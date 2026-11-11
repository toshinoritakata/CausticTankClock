// Debug / verification hooks: window.relax (stepping, metrics, god-ray + oblique tests, benchmarks) and window.clock (set/advance time).
// Not used by the app itself; kept for scripted checks from the console. All state comes in through K (see app.js):
// sub-modules never import each other (a static import would lose app.js's ?v= cache-busting query), app.js hands them what they need.

const corr = (a, b) => { const n = a.length; let ma = 0, mb = 0; for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; } ma /= n; mb /= n; let sab = 0, saa = 0, sbb = 0; for (let i = 0; i < n; i++) { const x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; } return sab / Math.sqrt(saa * sbb); };
const stat = a => { let m = 0, s2 = 0, mx = -1e30, mn = 1e30; for (let i = 0; i < a.length; i++) { m += a[i]; s2 += a[i] * a[i]; mx = Math.max(mx, a[i]); mn = Math.min(mn, a[i]); } m /= a.length; return { mean: m, std: Math.sqrt(Math.max(0, s2 / a.length - m * m)), max: mx, min: mn }; };
const grayOf = (cv, N) => { const d = cv.getContext('2d').getImageData(0, 0, N, N).data, g = new Float64Array(N * N); for (let i = 0; i < N * N; i++) g[i] = (0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2]) / 255; return g; };
const relRms = (a, b) => { let ma = 0, mb = 0; const n = a.length; for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; } ma /= n; mb /= n; let e = 0, v = 0; for (let i = 0; i < n; i++) { e += ((a[i] - ma) - (b[i] - mb)) ** 2; v += (b[i] - mb) ** 2; } return Math.sqrt(e / (v + 1e-30)); };

export function installDebug(K) {
  const { device, $, rd, app, S, R, clock, cpuMod, lightParams, rb, CD, fitCanvas, size, makeCtx, ensureVol, writeGod, encodeForward, encodeRender, writeUniforms,
    dynP, planSteps, dtSafe, writeDyn, readStats, solveStar, nextLetter, disturb, initSim, setN, setPlay, curLetter, uploadGlyph, paramsFinal, writeParams, amb } = K;
  const { clk, clockOn, showSec, shownStr } = clock;
  const { encodeSolve, encodeSteps, encodeCaustic, encodeVolume } = K.passes;
  const MAP_READ = GPUBufferUsage.MAP_READ;

  const frameOnce = async (render = true) => {
    const c = S.ctx; fitCanvas(); const enc = device.createCommandEncoder(), ps = enc.beginComputePass(); encodeForward(ps, c); ps.end();
    if (render) { writeUniforms(); encodeRender(enc, c); } device.queue.submit([enc.finish()]); await device.queue.onSubmittedWorkDone();
  };
  async function readBuf(src, stage) {
    const c = S.ctx, n = c.N * c.N, enc = device.createCommandEncoder(); enc.copyBufferToBuffer(src, 0, stage, 0, n * 4); device.queue.submit([enc.finish()]);
    await stage.mapAsync(GPUMapMode.READ); const a = new Float32Array(stage.getMappedRange().slice(0)); stage.unmap(); return a;
  }
  // read the 3D god-ray volume (or one z slice) into a Float32Array
  async function readVolume(g, origin, extent) {
    const st = device.createBuffer({ size: extent[0] * extent[1] * extent[2] * 4, usage: MAP_READ | CD }), enc = device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: g.vol, origin }, { buffer: st, bytesPerRow: g.M * 4, rowsPerImage: g.M }, extent); device.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ); const a = new Float32Array(st.getMappedRange().slice(0)); st.unmap(); st.destroy(); return a;
  }
  // design h* then the forward caustic of h* alone (e = 0); P = alternative pipeline set (old solver)
  async function eqRun(c, p, P = c.P) {
    writeParams(c, p); const cc = { N: c.N, bg: c.bg, P }, enc = device.createCommandEncoder(), ps = enc.beginComputePass();
    encodeSolve(ps, cc, p); ps.end(); enc.copyBufferToBuffer(c.B.field, 0, c.B.hT, 0, c.N * c.N * 4);
    const p2 = enc.beginComputePass(); encodeCaustic(p2, cc, P); p2.end();
    device.queue.submit([enc.finish()]); const nn = c.N * c.N * 4; return { h: await rb(c.B.field, nn), C: await rb(c.B.C, nn) };
  }
  const oldCache = {};
  async function oldPipes(c) {
    if (oldCache[c.N]) return oldCache[c.N];
    return oldCache[c.N] = await K.createOldPipelines(c);
  }

  const rx = window.relax = {
    state: () => ({ t: R.t, tl: R.tl, idx: R.idx, letter: curLetter(), resid: R.resid, rmsStar: R.rmsStar, emax: R.emax, nan: R.nan, playing: R.playing, holdT: R.holdT, nSub: R.nSub, hSub: R.hSub, slowed: R.slowed, dtSafe: dtSafe(dynP(), S.N), N: S.N }),
    pause: (v = true) => setPlay(!v),
    residual: async () => readStats(),
    // deterministic stepping with fixed substep dt (no auto letter switching); T seconds of simulated time
    step: async (T, dt) => {
      const c = S.ctx, P = dynP(); dt = dt ?? dtSafe(P, c.N) * 0.98; R.playing = false; let n = Math.round(T / dt);
      while (n > 0) { const m = Math.min(n, 48); writeDyn(c, P, dt, R.t); const enc = device.createCommandEncoder(), ps = enc.beginComputePass(); encodeSteps(ps, c, m); ps.end(); device.queue.submit([enc.finish()]); R.t += m * dt; R.tl += m * dt; n -= m; }
      await device.queue.onSubmittedWorkDone(); return R.t;
    },
    frame: frameOnce,
    getC: async () => { await frameOnce(false); return readBuf(S.ctx.B.C, S.ctx.stageB); },
    getH: async () => { await frameOnce(false); return readBuf(S.ctx.B.hT, S.ctx.stageB); },
    getE: async () => { const c = S.ctx; return readBuf(c.E[c.cur], c.stageE); },
    refC: async () => {   // caustic of h* alone (e = 0)
      const c = S.ctx, enc = device.createCommandEncoder(); enc.copyBufferToBuffer(c.B.field, 0, c.B.hT, 0, c.N * c.N * 4);
      const ps = enc.beginComputePass(); encodeCaustic(ps, c); ps.end(); device.queue.submit([enc.finish()]);
      return readBuf(c.B.C, c.stageB);
    },
    corr, stat,
    metrics: async ref => { const C = await rx.getC(), s = stat(C), r = stat(ref); return { corr: corr(C, ref), cv: s.std / s.mean, cvRatio: (s.std / s.mean) / (r.std / r.mean), res: (await readStats()).resid }; },
    // sample metrics at listed times (s since the last init/retarget) by stepping with fixed dt
    curve: async (times, dt) => {
      const ref = await rx.refC(), out = []; const t0 = R.tl;
      for (const T of times) { await rx.step(T - (R.tl - t0), dt); const m = await rx.metrics(ref); out.push({ t: +(R.tl - t0).toFixed(2), ...m }); } return out;
    },
    // long-run NaN / energy check
    soak: async (T, dt, every = 1) => { const out = []; for (let t = 0; t < T; t += every) { await rx.step(every, dt); const s = await readStats(); out.push([+(t + every).toFixed(1), +s.resid.toFixed(5), +s.emax.toExponential(2), s.nan]); } return out; },
    replay: async seed => { R.playing = false; await initSim(seed); },
    disturb, setLetter: i => nextLetter(i),
    set: o => { for (const k in o) { const el = $(k); if (el.type === 'checkbox') el.checked = !!o[k]; else el.value = o[k]; } },
    setN: async n => { n = Math.max(128, n); $('N').value = n; await setN(n); },   // UI select only offers 128/256/512
    // ----- god rays hooks -----
    godrays: v => { if (v !== undefined) $('god').checked = !!v; return $('god').checked; },
    volumeStats: async () => {
      const c = S.ctx, was = $('god').checked; $('god').checked = true; await frameOnce(false); $('god').checked = was;
      const g = c.god, n = g.M * g.M * g.NZ, a = await readVolume(g, undefined, [g.M, g.M, g.NZ]);
      const sl = [], mm = g.M * g.M; let nan = 0;
      for (let z = 0; z < g.NZ; z++) { const q = a.subarray(z * mm, (z + 1) * mm); for (let i = 0; i < mm; i++) if (!isFinite(q[i])) nan++; const t = stat(q); sl.push({ z: +(rd('D') * z / (g.NZ - 1)).toFixed(3), mean: +t.mean.toFixed(4), std: +t.std.toFixed(3), max: +t.max.toFixed(2), min: +t.min.toFixed(3) }); }
      return { M: g.M, NZ: g.NZ, bytes: n * 4, nan, slices: sl, data: a };
    },
    checkSliceD: async () => {
      const c = S.ctx, was = $('god').checked; $('god').checked = true; await frameOnce(false); $('god').checked = was;
      const g = c.god, M = g.M, sub = c.N / M, sl = await readVolume(g, [0, 0, g.NZ - 1], [M, M, 1]);
      const C = await readBuf(c.B.C, c.stageB), pt = new Float32Array(M * M), bl = new Float32Array(M * M);
      for (let y = 0; y < M; y++) for (let x = 0; x < M; x++) { pt[y * M + x] = C[y * sub * c.N + x * sub]; let a = 0; for (let j = 0; j < sub; j++) for (let i = 0; i < sub; i++) a += C[(y * sub + j) * c.N + x * sub + i]; bl[y * M + x] = a / (sub * sub); }
      return { corrPoint: corr(sl, pt), corrBlock: corr(sl, bl), sliceStat: stat(sl), floorStat: stat(pt) };
    },
    benchParts: async (K_ = 60) => {
      const c = S.ctx, P = dynP(), pl = planSteps(1 / 60, P, c.N), was = $('god').checked; fitCanvas(); ensureVol(c); writeUniforms(); writeDyn(c, P, pl.h, 0);
      const t = async (fn, n) => { for (let i = 0; i < 3; i++) fn(); await device.queue.onSubmittedWorkDone(); const t0 = performance.now(); for (let i = 0; i < n; i++) fn(); await device.queue.onSubmittedWorkDone(); return (performance.now() - t0) / n; };
      const full = () => { const enc = device.createCommandEncoder(), ps = enc.beginComputePass(); encodeSteps(ps, c, pl.n); encodeForward(ps, c); ps.end(); writeUniforms(); encodeRender(enc, c); device.queue.submit([enc.finish()]); };
      const out = { N: c.N, M: c.god.M, NZ: c.god.NZ, w: size.w, h: size.h, steps: rd('rsteps') };
      $('god').checked = false; out.frameOff = await t(full, K_);
      $('god').checked = true; out.frameOn = await t(full, K_);
      writeGod(c); out.volumeBuild = await t(() => { const enc = device.createCommandEncoder(), ps = enc.beginComputePass(); encodeVolume(ps, c); ps.end(); device.queue.submit([enc.finish()]); }, K_);
      out.raymarchApprox = out.frameOn - out.frameOff - out.volumeBuild; $('god').checked = was; return out;
    },
    resolve: () => solveStar(true),
    getT: async () => rb(S.ctx.B.T, S.ctx.N * S.ctx.N * 4), getStar: async () => rb(S.ctx.B.field, S.ctx.N * S.ctx.N * 4),
    // ----- oblique tests: GPU vs CPU float64 reference, regression vs the vertical solver (solver_old.js, the original vertical-light solver) -----
    setLight: async o => { rx.set(o); await new Promise(r => setTimeout(r, 400)); await device.queue.onSubmittedWorkDone(); },
    compareOblique: async (opts = {}) => {
      const N = opts.N || 128, pairs = opts.pairs || [[90, 0, true], [65, 30, true], [60, 45, true], [45, 135, true], [35, 200, true], [65, 30, false], [45, 135, false]];
      const was = R.playing; R.playing = false; await device.queue.onSubmittedWorkDone(); const rows = [];
      const c = await makeCtx(N), keep = S.paused; S.paused = true;
      try {
        for (const [el, az, center] of pairs) {
          const p = { ...paramsFinal(), D: opts.D || 0.6, el, az, center, wrap: !!opts.wrap, cast: opts.cast ?? 0.5, fit: true };
          uploadGlyph(c, p, opts.letter || 'A'); const gA = grayOf(c.cv[0], N);
          const g = await eqRun(c, p), L = lightParams(el, az, p.D, center, p.wrap), cpu = cpuMod.cpuRun(gA, gA, N, p, L);
          rows.push({ el, az, center, hErr: relRms(g.h, cpu.h), CErr: relRms(g.C, cpu.C), CCorr: +corr(g.C, cpu.C).toFixed(4), s: L.s.map(v => +v.toFixed(3)) });
        }
      } finally { S.paused = keep; R.playing = was; }
      return rows;
    },
    regression90: async (N = 256) => {
      const was = R.playing; R.playing = false; await device.queue.onSubmittedWorkDone(); const c = await makeCtx(N), keep = S.paused; S.paused = true;
      try {
        const p = { ...paramsFinal(), D: 1, el: 90, az: 0, center: true, wrap: false, cast: 0 };
        uploadGlyph(c, p, 'A'); const nw = await eqRun(c, p), od = await eqRun(c, p, await oldPipes(c));
        return { N, hRelRms: relRms(nw.h, od.h), CRelRms: relRms(nw.C, od.C), CCorr: corr(nw.C, od.C), Cmean: stat(nw.C).mean };
      } finally { S.paused = keep; R.playing = was; }
    },
    bench: async (K_ = 60, frameDt = 1 / 60) => {
      const c = S.ctx, P = dynP(), pl = planSteps(frameDt, P, c.N); fitCanvas(); writeUniforms(); writeDyn(c, P, pl.h, 0);
      const run = async n => { const t0 = performance.now(); for (let i = 0; i < n; i++) { const enc = device.createCommandEncoder(), ps = enc.beginComputePass(); encodeSteps(ps, c, pl.n); encodeForward(ps, c); ps.end(); encodeRender(enc, c); device.queue.submit([enc.finish()]); } await device.queue.onSubmittedWorkDone(); return (performance.now() - t0) / n; };
      await run(5); const ms = await run(K_); return { N: c.N, substeps: pl.n, subDtMs: pl.h * 1000, w: size.w, h: size.h, msPerFrame: ms, fps: 1000 / ms };
    },
    ambient: { state: amb.state, ratio: async () => { const s = await readStats(); return { ratio: Math.sqrt(s.e2) / R.rmsStar, nan: s.nan }; } },
  };

  window.clock = {
    setTime: s => { const m = /^(\d{1,2}):?(\d{2})(?::?(\d{2}))?$/.exec(s); if (!m) throw new Error('HH:MM[:SS]'); const d = new Date(); d.setHours(+m[1], +m[2], +(m[3] || 0), 0); clk.off = d.getTime() - Date.now(); return shownStr(); },
    advance: min => { clk.off += min * 60000; return shownStr(); },
    state: () => ({ on: clockOn(), time: shownStr(), key: clk.key, ticks: clk.ticks, offsetMs: clk.off, layout: $('clockLayout').value, sec: showSec(), h12: $('clock12').value, N: S.N, resid: R.resid, nan: R.nan, fps: app.fps }),
  };
}
