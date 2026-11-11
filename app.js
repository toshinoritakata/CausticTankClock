// CausticTankClock: the water surface is a real wave field h = h* + e.  h* (designed surface) comes from the live WebGPU
// Monge-Ampere solver (solver.js, run only when the letter/clock/light settings change); e obeys a damped wave equation with restoring
// force towards 0 (relax.js), so lively waves settle into h*.  The forward splat+blur caustic and the mesh are computed from h every frame.
// (solver.js / relax.js descend from an earlier standalone experiment; solver_old.js is that vertical-light solver, kept for regression checks.)
//
// This file is the orchestrator. Modules (all leaf modules, loaded below with this file's ?v= cache-busting query, so none can be stale-cached):
//   gpu.js (device, pipelines, buffers)  passes.js (compute encoders)  glyph.js (letter / clock texture)  clock.js (time + glyph key)
//   dynamics.js (stability, ambient forcing)  camera.js  ui.js (panel, keys, HUD)  params.js (localStorage)  debug.js (window.relax / window.clock)
const $ = id => document.getElementById(id);
const V = new URL(import.meta.url).searchParams.get('v') || Date.now();
const load = name => import(`./${name}.js?v=${V}`);
const [solverMod, renderMod, relaxMod, godMod, obliqueMod, cpuMod, gpu, passes, glyph, clockMod, dyn, cameraMod, uiMod, paramsMod, debugMod] =
  await Promise.all(['solver', 'render', 'relax', 'godrays', 'oblique', 'cpu', 'gpu', 'passes', 'glyph', 'clock', 'dynamics', 'camera', 'ui', 'params', 'debug'].map(load));
const { lightParams, NW } = obliqueMod;
const { genIC, mulberry32 } = relaxMod;
const { encodeSolve, encodeSteps, encodeCaustic, encodeVolume, submitKick } = passes;
const qp = new URLSearchParams(location.search);

const MAX_FRAME_DT = 0.05;        // s: longest frame step fed to the simulation
const PRE_FINAL = 1.5;            // solver pre-blur at N = 512
const SOLVER_BETA = 0.7;
const STATS_EVERY_FRAMES = 6;     // GPU stats readback interval
const BLOWUP_RESID = 50;          // residual |e|/|h*| above which the sim is re-initialised
const FPS_PERIOD_MS = 500;
const MAX_LETTER_S = 60;          // letter list: advance after this long even if not settled
const KICK_RING_AMP = 4;          // clock tick ring amplitude, x rms(h*) x 'tick kick'
const KICK_SECONDS_SCALE = 0.35;  // ... scaled down when seconds are shown
const MIN_SETTLE_S = 0.5;
const DEFAULT_N = 256;

async function main() {
  let app;
  const canvas = $('view');
  const { device, gctx, fmt } = await gpu.initDevice(canvas, msg => { app.lastError = msg; });
  app = window.tank = { device, lastError: null, fps: 0, frames: 0 };   // debug API (tank.cam, tank.preset, tank.state, ...)
  const rd = id => +$(id).value;

  const renderer = await gpu.createRenderer({ device, gctx, fmt, canvas, qp, SCENE_WGSL: renderMod.SCENE_WGSL, BG_WGSL: renderMod.BG_WGSL });
  const { fitCanvas, size } = renderer;
  const { makeCtx, ensureVol: ensureVolFor } = gpu.createContexts({ device, shaders: { computeWGSL: solverMod.computeWGSL, RELAX_WGSL: relaxMod.RELAX_WGSL, godWGSL: godMod.godWGSL }, renderer });
  const ensureVol = c => ensureVolFor(c, +$('volres').value, rd('nz'));
  const rb = (buf, bytes) => gpu.readBuffer(device, buf, bytes);

  // ---------- state ----------
  // S: current context / grid / light; R: simulation + relax state
  const S = { ctx: null, N: 512, paused: true, t0: performance.now() };
  const R = { t: 0, tl: 0, idx: 0, playing: true, holdT: 0, resid: 1, e2: 0, h2: 1e-4, emax: 0, nan: 0, rmsStar: 0.01, busy: false, statBusy: false, slowed: false, nSub: 0, hSub: 0, listStr: null };
  app.state = S;
  app.stretchLaw = 'sin';   // glyph cast-shadow stretch law, see glyph.js castWarp
  const clock = clockMod.createClock($);
  const { clk, clockOn, showSec, clockKey } = clock;
  const letters = () => { const L = $('textList').value.split(/[,，]/).map(x => x.trim()).filter(Boolean); return L.length ? L : [' ']; };
  const curLetter = () => { if (clockOn()) { if (!clk.key) clk.key = clockKey(); return clk.key; } return letters()[R.idx % letters().length]; };
  const meshM = c => $('decouple').checked ? Math.min(c.N, Math.max(2, +$('mesh').value || c.N)) : c.N;

  // ---------- dynamics ----------
  const dynP = () => dyn.dynParams(rd, S.N, clockOn() && showSec());
  const amb = dyn.createAmbient({ device, $, rd, mulberry32 });
  const writeDyn = (c, P, h, t) => device.queue.writeBuffer(c.rU, 0, new Float32Array([h, P.c2, dyn.kappaEff(P, t, rd('ramp')), P.gamma, P.nu, P.s, c.N * c.N, 0]));

  // ---------- solver (h*) ----------
  const paramsFinal = () => ({ D: rd('D'), contrast: rd('contrast'), preblur: PRE_FINAL * S.N / 512, blur: 2 * rd('rblur') * S.N / 512, iters: rd('iters'), beta: SOLVER_BETA, sub: rd('subray'), morph: 0, el: rd('elev'), az: rd('az'), center: $('center').checked, wrap: $('wrap').checked, cast: rd('cast'), fit: $('fit').checked });
  function writeParams(c, p) {
    const k = p.D * (1 - 1 / NW), nn = c.N * c.N;
    const L = lightParams(p.el, p.az, p.D, p.center, p.wrap); S.L = L;
    device.queue.writeBuffer(c.B.params, 0, new Float32Array([p.D, k, p.contrast, p.morph, p.preblur, p.blur, p.sub, NW, p.beta, 0, 0, 0,
      L.K[0], L.K[1], L.K[2], L.shDes[0], L.shDes[1], L.sOverD[0], L.sOverD[1], L.di[0], L.di[1], L.di[2], L.fadeW, 0]));
    device.queue.writeBuffer(c.B.mixIter, 0, new Float32Array([p.beta / nn, 1 - p.beta, 0, 0]));
  }
  function uploadGlyph(c = S.ctx, q = paramsFinal(), text = curLetter()) {
    const N = c.N; if (c.rawKey !== text + N) { glyph.drawGlyph(c.raw[0], text, N); c.rawKey = text + N; }
    glyph.castWarp(c, q, app.stretchLaw); device.queue.copyExternalImageToTexture({ source: c.cv[0] }, { texture: c.texA }, [N, N]);
  }
  // solve h* for the current letter/settings (only on change). retarget: keep the present surface, e += h*_old - h*_new.
  function solveStar(retarget) {
    const c = S.ctx, p = paramsFinal(); if (clockOn()) clk.last = curLetter(); uploadGlyph(c, p); writeParams(c, p);
    const enc = device.createCommandEncoder();
    if (retarget) enc.copyBufferToBuffer(c.B.field, 0, c.hp, 0, c.N * c.N * 4);
    const ps = enc.beginComputePass(); encodeSolve(ps, c, p);
    if (retarget) { ps.setPipeline(c.rp.retarget); ps.setBindGroup(0, c.bgP[1 - c.cur]); ps.dispatchWorkgroups(c.N / 8, c.N / 8); }
    ps.end(); device.queue.submit([enc.finish()]);
  }

  // ---------- god rays + forward pass ----------
  const godOn = () => $('god').checked;
  function writeGod(c) {
    const g = c.god, ab = new ArrayBuffer(64), f = new Float32Array(ab), u = new Uint32Array(ab), sig = 2 * g.M / 512;   // blur sigma: 2 cells at 512
    f[0] = rd('D'); f[1] = 1 / NW; u[2] = g.M; u[3] = g.NZ; u[4] = g.sub; f[5] = c.N / g.M; f[6] = Math.max(sig, 0.3); u[7] = Math.ceil(3 * f[6]); f[8] = rd('rnoise'); f[9] = (performance.now() - S.t0) / 1000;
    const L = S.L || lightParams(90, 0, f[0]); f[10] = L.di[0]; f[11] = L.di[1]; f[12] = L.di[2]; f[13] = L.fadeW;
    device.queue.writeBuffer(g.gU, 0, ab);
  }
  // forward pass: hT = h* + e ; grad ; splat ; blur -> C ; optional god-ray volume
  function encodeForward(pass, c) {
    pass.setPipeline(c.rp.compose); pass.setBindGroup(0, c.bgP[c.cur]); pass.dispatchWorkgroups(c.N / 8, c.N / 8);
    encodeCaustic(pass, c);
    if (godOn()) { ensureVol(c); writeGod(c); encodeVolume(pass, c); }
  }

  // ---------- stats readback (mean e^2, mean h*^2, max|e|, non-finite count) ----------
  let statChain = Promise.resolve();
  function readStats() {
    const run = async () => {
      const c = S.ctx, enc = device.createCommandEncoder(), ps = enc.beginComputePass();
      ps.setPipeline(c.rp.reduce); ps.setBindGroup(0, c.bgP[c.cur]); ps.dispatchWorkgroups(1); ps.end();
      enc.copyBufferToBuffer(c.rstats, 0, c.statStage, 0, 16); device.queue.submit([enc.finish()]);
      await c.statStage.mapAsync(GPUMapMode.READ); const a = new Float32Array(c.statStage.getMappedRange().slice(0)); c.statStage.unmap();
      R.e2 = a[0]; R.h2 = a[1]; R.emax = a[2]; R.nan = a[3]; R.resid = Math.sqrt(a[0] / Math.max(a[1], 1e-30)); R.rmsStar = Math.sqrt(a[1]);
      return { e2: a[0], h2: a[1], emax: a[2], nan: a[3], resid: R.resid };
    };
    return statChain = statChain.then(run, run);
  }

  // ---------- initial condition / disturbance / clock stirring ----------
  const pickSeed = () => $('det').checked ? rd('seed') : Math.floor(Math.random() * 1e9);
  async function initSim(seed) {
    R.busy = true; const c = S.ctx;
    R.idx = 0; R.listStr = $('textList').value; solveStar(false); await readStats();
    const P = dynP(), ic = genIC(c.N, seed ?? pickSeed(), rd('amp') * R.rmsStar, P);
    device.queue.writeBuffer(c.E[c.cur], 0, ic.e); device.queue.writeBuffer(c.Vb[c.cur], 0, ic.v);
    R.t = 0; R.tl = 0; R.holdT = 0; R.resid = rd('amp'); R.busy = false;
  }
  function disturb() {   // velocity-only kick (surface stays continuous), amplitude = amp * rms(h*)
    const c = S.ctx, ic = genIC(c.N, pickSeed(), rd('amp') * R.rmsStar, dynP());
    device.queue.writeBuffer(c.kV, 0, ic.v); submitKick(device, c); R.holdT = 0; R.tl = 0;
  }
  // clock water stirring: velocity-only ring from the centre (tick) + faint random shimmer, all through the same addKick path
  function kickBuf(c) {
    if (c.kk) return c.kk; const N = c.N, ring = new Float32Array(N * N); let m = 0;
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      const dx = (i + 0.5) / N - 0.5, dy = (j + 0.5) / N - 0.5, r = Math.hypot(dx, dy), q = (r - 0.13) / 0.08;
      const v = Math.exp(-q * q) * Math.cos(2 * Math.PI * (r - 0.13) / 0.14); ring[j * N + i] = v; m += v;
    }
    m /= N * N; let s = 0; for (let i = 0; i < ring.length; i++) { ring[i] -= m; s += ring[i] * ring[i]; } s = Math.sqrt(s / ring.length) || 1; for (let i = 0; i < ring.length; i++) ring[i] /= s;   // zero mean, unit rms
    return c.kk = { ring, out: new Float32Array(N * N) };
  }
  function applyKick(c, src, amp) {
    const o = kickBuf(c).out; for (let i = 0; i < o.length; i++) o[i] = src[i] * amp;
    device.queue.writeBuffer(c.kV, 0, o); submitKick(device, c);
  }
  function clockTick() {
    const c = S.ctx; R.tl = 0; R.resid = 1; R.holdT = 0; clk.ticks++; solveStar(true);
    const k = rd('kick') * (showSec() ? KICK_SECONDS_SCALE : 1); if (k > 0) applyKick(c, kickBuf(c).ring, k * KICK_RING_AMP * R.rmsStar);
  }
  function nextLetter(to) {
    if (letters().length <= 1) { disturb(); return; }
    R.idx = to != null ? to : (R.idx + 1) % letters().length; R.tl = 0; R.holdT = 0; R.resid = 1; solveStar(true);
  }

  // ---------- camera, render ----------
  const camCtl = cameraMod.createCamera({ $, rd, canvas, app });
  const { cam } = camCtl;
  const encodeRender = (enc, c) => renderer.encodeRender(enc, c, meshM(c));
  function writeUniforms() {
    const D = rd('D'), asp = size.w / size.h, { eye, lk, vp } = cameraMod.cameraView(cam, D, asp), fov = cameraMod.FOV;
    const u = new Float32Array(64);
    u.set(vp, 0);
    u.set([eye[0], eye[1], eye[2], (performance.now() - S.t0) / 1000], 16);
    u.set([-lk.z[0], -lk.z[1], -lk.z[2], Math.tan(fov / 2)], 20);
    u.set([lk.x[0], lk.x[1], lk.x[2], asp], 24);
    u.set([lk.y[0], lk.y[1], lk.y[2], 0], 28);
    u.set([D, rd('relief'), rd('expo'), S.N], 32);
    const gl = rd('glow');
    u.set([1.0 + 0.8 * gl, $('glass').checked ? 1 : 0, size.w, size.h], 36);
    u.set([gl, 0, 0, 0], 40);
    u.set([godOn() ? 1 : 0, rd('rgain'), rd('rsig'), rd('rg')], 44);
    u.set([rd('rsteps'), rd('rnoise'), S.ctx.god.NZ || 24, 0], 48);
    const L = S.L || lightParams(90, 0, D), sw = L.sunWorld, ce = Math.cos(L.el * cameraMod.RAD), eta = 1 / NW;
    u.set([sw[0], sw[1], sw[2], 0], 52);
    const wy = -Math.sqrt(1 - eta * eta * ce * ce), wl = Math.hypot(eta * ce, wy);   // refracted travel direction at a flat surface (world x, up, z); world z == solver y
    u.set([eta * L.di[0] / wl, wy / wl, eta * L.di[1] / wl, 0], 56);
    u.set([meshM(S.ctx), $('flat').checked ? 1 : 0, $('csamp').value === 'nearest' ? 1 : 0, 0], 60);
    device.queue.writeBuffer(renderer.ubuf, 0, u);
  }

  // wave resolution: N = surface vertices = rays traced = caustic resolution. On change the new h* is solved and e is resampled from the old surface (h_old - h*_new): no pop.
  const resamp = (src, No, Nn) => {
    const o = new Float32Array(Nn * Nn), k = No / Nn;
    for (let y = 0; y < Nn; y++) {
      const fy = y * k, y0 = Math.floor(fy), ay = fy - y0, ya = y0 % No, yb = (y0 + 1) % No;
      for (let x = 0; x < Nn; x++) {
        const fx = x * k, x0 = Math.floor(fx), ax = fx - x0, xa = x0 % No, xb = (x0 + 1) % No;
        o[y * Nn + x] = (src[ya * No + xa] * (1 - ax) + src[ya * No + xb] * ax) * (1 - ay) + (src[yb * No + xa] * (1 - ax) + src[yb * No + xb] * ax) * ay;
      }
    }
    return o;
  };
  async function setN(N) {
    const old = S.ctx; S.paused = true; R.busy = true;
    let hOld = null, vOld = null, No = 0;
    if (old && old.N !== N) { No = old.N; const f = await rb(old.B.field, No * No * 4), e = await rb(old.E[old.cur], No * No * 4); vOld = await rb(old.Vb[old.cur], No * No * 4); hOld = f.map((v, i) => v + e[i]); }
    S.ctx = await makeCtx(N); ensureVol(S.ctx); S.N = N;
    if (!hOld) { await initSim(); }
    else {
      const c = S.ctx; solveStar(false); await device.queue.onSubmittedWorkDone();
      const fNew = await rb(c.B.field, N * N * 4), hr = resamp(hOld, No, N), e = hr.map((v, i) => v - fNew[i]);
      device.queue.writeBuffer(c.E[c.cur], 0, e); device.queue.writeBuffer(c.Vb[c.cur], 0, resamp(vOld, No, N));
      R.holdT = 0; await readStats();
    }
    R.busy = false; S.paused = false;
  }

  // ---------- panel, debug hooks, saved params ----------
  const ui = uiMod.initUI({ $, rd, qp, R, clock, app, solveStar, setN, initSim, nextLetter, disturb, curLetter });
  debugMod.installDebug({
    device, $, rd, app, S, R, clock, cpuMod, lightParams, rb, CD: gpu.CD, fitCanvas, size, makeCtx, ensureVol, writeGod, encodeForward, encodeRender, writeUniforms,
    dynP, planSteps: dyn.planSteps, dtSafe: dyn.dtSafe, writeDyn, readStats, solveStar, nextLetter, disturb, initSim, setN, setPlay: ui.setPlay, curLetter, uploadGlyph, paramsFinal, writeParams, amb, passes,
    // the pre-oblique vertical-light solver, loaded on demand (regression90 only)
    createOldPipelines: async c => {
      const { computeWGSL } = await load('solver_old'), mod = await gpu.compileWGSL(device, computeWGSL(c.N), 'solverOld' + c.N);
      return gpu.solverPipelines(device, mod, c.layout);
    },
  });
  paramsMod.initParams({ $, qp, canvas, app, cam, readTargetSliders: camCtl.readTargetSliders, snapCam: camCtl.snapCam,
    camRange: { PITCH_MIN: cameraMod.PITCH_MIN, PITCH_MAX: cameraMod.PITCH_MAX, DIST_MIN: cameraMod.DIST_MIN, DIST_MAX: cameraMod.DIST_MAX },
    onImported: () => { ui.syncMesh(); ui.drawCompass(); } });
  ui.syncMesh(); ui.drawCompass();

  // ---------- frame ----------
  let tPrev = performance.now(), frameCount = 0, tFps = performance.now(), fr = 0;
  function frame(now) {
    requestAnimationFrame(frame);
    const c = S.ctx; if (!c || S.paused) return;
    ensureVol(c);
    const dtR = Math.min(MAX_FRAME_DT, (now - tPrev) / 1000); tPrev = now; fr++;
    ui.updateValueLabels();
    camCtl.step(now, dtR);
    fitCanvas();
    if (clockOn()) { R.listStr = $('textList').value; if (!R.busy) { clk.key = clockKey(); if (clk.key !== clk.last) clockTick(); } }
    else if (R.listStr !== $('textList').value && !R.busy) { R.listStr = $('textList').value; nextLetter(0); }
    const enc = device.createCommandEncoder(), ps = enc.beginComputePass();
    let simDt = 0;
    if (R.playing && !R.busy) {
      const P = dynP(), pl = dyn.planSteps(dtR, P, c.N); R.slowed = pl.slowed; R.nSub = pl.n; R.hSub = pl.h;
      writeDyn(c, P, pl.h, R.t); simDt = pl.n * pl.h; amb.update(c, simDt, P, R.rmsStar); encodeSteps(ps, c, pl.n);
    }
    encodeForward(ps, c); ps.end();
    writeUniforms(); encodeRender(enc, c); device.queue.submit([enc.finish()]);
    if (R.playing && !R.busy) {
      R.t += simDt; R.tl += simDt;
      if (R.resid < rd('thr') && R.tl > MIN_SETTLE_S) R.holdT += dtR; else R.holdT = 0;
      if (!clockOn() && (R.holdT >= rd('hold') || R.tl > MAX_LETTER_S)) nextLetter();
    }
    if ((!isFinite(R.resid) || R.nan > 0 || R.resid > BLOWUP_RESID) && !R.busy) { console.warn('relax: blow-up detected, re-initialising'); initSim(); }
    if (fr % STATS_EVERY_FRAMES === 0 && !R.busy && !R.statBusy) { R.statBusy = true; readStats().catch(() => {}).finally(() => { R.statBusy = false; }); }
    ui.updateHud();
    app.frames++; frameCount++;
    if (now - tFps > FPS_PERIOD_MS) {
      app.fps = frameCount * 1000 / (now - tFps); frameCount = 0; tFps = now; const mm = meshM(c);
      $('fps').textContent = `${app.fps.toFixed(0)} fps  N=${S.N}  ${size.w}x${size.h}\nmesh ${mm}x${mm}  triangles: ${(2 * (mm - 1) ** 2).toLocaleString('en-US')}\nsubsteps ${R.nSub} x ${(R.hSub * 1000).toFixed(2)} ms` + (app.lastError ? '\nERROR ' + app.lastError : '');
    }
  }

  // start at ?n= (128 / 256 / 512) or the panel's N
  { const n0 = [128, 256, 512].includes(+qp.get('n')) ? +qp.get('n') : (+$('N').value || DEFAULT_N); $('N').value = n0; await setN(n0); }
  requestAnimationFrame(frame);
  app.ready = true;
}
main().catch(e => { document.getElementById('fps').textContent = 'ERROR: ' + e.message; console.error(e); });
