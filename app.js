// CausticTankClock: the water surface is a real wave field h = h* + e.  h* (designed surface) comes from the live WebGPU
// Monge-Ampere solver (solver.js, run only when letter/settings change); e obeys a damped wave equation with restoring force towards 0
// (relax.js), so lively waves settle into h*.  The forward splat+blur caustic and the mesh are computed from h every frame.
const $ = id => document.getElementById(id);
const V = Date.now();
const { computeWGSL } = await import('./solver.js?v=' + V);
const { SCENE_WGSL, BG_WGSL } = await import('./render.js?v=' + V);
const { RELAX_WGSL, genIC, mulberry32 } = await import('./relax.js?v=' + V);
const { godWGSL } = await import('./godrays.js?v=' + V);
const { lightParams, NW } = await import('./oblique.js?v=' + V);
const cpuMod = await import('./cpu.js?v=' + V);
const qp = new URLSearchParams(location.search);

// ---------- small mat4 helpers (column-major, WebGPU depth 0..1) ----------
// ---------- small mat4 helpers (column-major, WebGPU depth 0..1) ----------
const vsub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const vcross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const vnorm = a => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const vdot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
function perspective(fovy, asp, n, f) { const t = 1 / Math.tan(fovy / 2), m = new Float32Array(16); m[0] = t / asp; m[5] = t; m[10] = f / (n - f); m[11] = -1; m[14] = f * n / (n - f); return m; }
function lookAt(e, c, up) {
  const z = vnorm(vsub(e, c)), x = vnorm(vcross(up, z)), y = vcross(z, x);
  const m = new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -vdot(x, e), -vdot(y, e), -vdot(z, e), 1]);
  return { m, x, y, z };
}
function mul(a, b) { const o = new Float32Array(16); for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) { let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k]; o[c * 4 + r] = s; } return o; }
async function main() {
  if (!navigator.gpu) throw new Error('WebGPU not available');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('no adapter');
  const device = await adapter.requestDevice({ requiredFeatures: adapter.features.has('float32-filterable') ? ['float32-filterable'] : [] });
  const app = window.tank = { device, lastError: null, fps: 0, frames: 0 };
  device.addEventListener('uncapturederror', e => { console.error('WebGPU uncaptured error: ' + e.error.message); app.lastError = e.error.message; });
  device.lost.then(i => console.error('device lost: ' + i.message));
  const canvas = $('view'), gctx = canvas.getContext('webgpu');
  const fmt = navigator.gpu.getPreferredCanvasFormat();
  gctx.configure({ device, format: fmt, alphaMode: 'opaque' });
  const ST = GPUBufferUsage.STORAGE, CS = GPUBufferUsage.COPY_SRC, CD = GPUBufferUsage.COPY_DST, UN = GPUBufferUsage.UNIFORM;
  const SAMPLES = 4;

  async function module(code, label) {
    const m = device.createShaderModule({ code });
    const info = await m.getCompilationInfo();
    for (const x of info.messages) console.log(`WGSL ${label} ${x.type} line ${x.lineNum}: ${x.message}`);
    if (info.messages.some(x => x.type === 'error')) throw new Error('WGSL compile error in ' + label);
    return m;
  }

  // ---------- render pipelines ----------
  const sceneMod = await module(SCENE_WGSL, 'scene'), bgMod = await module(BG_WGSL, 'bg');
  const rbgl = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    ...[1, 2, 3, 4].map(b => ({ binding: b, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } })),
    { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '3d' } },
    { binding: 6, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
  ] });
  const vsamp = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge' });
  const rlayout = device.createPipelineLayout({ bindGroupLayouts: [rbgl] });
  const blend = { color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } };
  const mkRP = (mod, vs, fs, o = {}) => device.createRenderPipeline({
    layout: rlayout, vertex: { module: mod, entryPoint: vs }, fragment: { module: mod, entryPoint: fs, targets: [{ format: fmt, blend: o.blend }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: o.dw !== false, depthCompare: o.cmp || 'less' },
    multisample: { count: SAMPLES },
  });
  const RP = {
    bg: mkRP(bgMod, 'vs', 'fs', { dw: false, cmp: 'always' }),
    table: mkRP(sceneMod, 'vsTable', 'fsTable'), floor: mkRP(sceneMod, 'vsFloor', 'fsFloor'),
    water: mkRP(sceneMod, 'vsWater', 'fsWater', { blend }), glass: mkRP(sceneMod, 'vsGlass', 'fsGlass', { blend, dw: false }),
  };
  const ubuf = device.createBuffer({ size: 256 + 96, usage: UN | CD });
  let msaaTex = null, depthTex = null, rw = 0, rh = 0;
  function fitCanvas() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const fw = +qp.get('w'), fh = +qp.get('h');
    const w = fw || Math.max(64, Math.round(canvas.clientWidth * dpr)), h = fh || Math.max(64, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h || !msaaTex) {
      canvas.width = w; canvas.height = h; rw = w; rh = h;
      msaaTex?.destroy(); depthTex?.destroy();
      msaaTex = device.createTexture({ size: [w, h], sampleCount: SAMPLES, format: fmt, usage: GPUTextureUsage.RENDER_ATTACHMENT });
      depthTex = device.createTexture({ size: [w, h], sampleCount: SAMPLES, format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });
    }
  }

  // ---------- per-N context: solver + relax buffers ----------
  const ctxCache = {};
  async function makeCtx(N) {
    if (ctxCache[N]) return ctxCache[N];
    const mod = await module(computeWGSL(N), 'solver' + N);
    const entries = [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ...[2, 3, 4, 5, 6, 7, 8].map(b => ({ binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } })),
      { binding: 9, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 10, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ];
    const bgl = device.createBindGroupLayout({ entries });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
    const pipe = (entryPoint, constants) => device.createComputePipeline({ layout, compute: { module: mod, entryPoint, constants } });
    const P = {};
    for (const e of ['rowsFwd', 'rowsInv', 'mixTex', 'prepT', 'normT', 'initT', 'maWarp', 'grad', 'clearAcc', 'accumToC', 'accumCopy', 'splat']) P[e] = pipe(e);
    P.colsPoisson = pipe('colsMul', { MODE: 0 }); P.colsPre = pipe('colsMul', { MODE: 1 }); P.colsC = pipe('colsMul', { MODE: 2 });
    P.red0 = pipe('reduce', { MODE: 0 }); P.red1 = pipe('reduce', { MODE: 1 });
    const nn = N * N, mk = (size, usage) => device.createBuffer({ size, usage });
    const B = {
      params: mk(96, UN | CD), mixPre: mk(16, UN | CD), mixFirst: mk(16, UN | CD), mixIter: mk(16, UN | CD),
      cbuf: mk(nn * 8, ST | CS | CD), field: mk(nn * 4, ST | CS | CD), T: mk(nn * 4, ST | CS | CD),
      hx: mk(nn * 4, ST | CS), hy: mk(nn * 4, ST | CS), accum: mk(nn * 4, ST | CS), stats: mk(16, ST | CS),
      C: mk(nn * 4, ST | CS | CD), tw: mk((N / 2 + 1 + N) * 8, ST | CD), hT: mk(nn * 4, ST | CS | CD),
    };   // B.field = h* (solver output); B.hT = h* + e (what is splatted and meshed)
    const twd = new Float32Array((N / 2 + 1 + N) * 2);
    for (let m = 0; m <= N / 2; m++) { twd[2 * m] = Math.cos(2 * Math.PI * m / N); twd[2 * m + 1] = -Math.sin(2 * Math.PI * m / N); }
    for (let x = 0; x < N; x++) twd[2 * (N / 2 + 1 + x)] = 4 * Math.sin(Math.PI * x / N) ** 2;
    device.queue.writeBuffer(B.tw, 0, twd);
    const texDesc = { size: [N, N], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT };
    const texA = device.createTexture(texDesc), texB = device.createTexture(texDesc);
    const bgFor = (mixBuf, fieldBuf) => device.createBindGroup({ layout: bgl, entries: [
      { binding: 0, resource: { buffer: B.params } }, { binding: 1, resource: { buffer: mixBuf } },
      { binding: 2, resource: { buffer: B.cbuf } }, { binding: 3, resource: { buffer: fieldBuf } },
      { binding: 4, resource: { buffer: B.T } }, { binding: 5, resource: { buffer: B.hx } }, { binding: 6, resource: { buffer: B.hy } },
      { binding: 7, resource: { buffer: B.accum } }, { binding: 8, resource: { buffer: B.stats } },
      { binding: 9, resource: texA.createView() }, { binding: 10, resource: texB.createView() },
      { binding: 11, resource: { buffer: B.tw } }] });
    const bg = { pre: bgFor(B.mixPre, B.field), first: bgFor(B.mixFirst, B.field), iter: bgFor(B.mixIter, B.field), c: bgFor(B.mixPre, B.C), R: bgFor(B.mixIter, B.hT) };
    device.queue.writeBuffer(B.mixPre, 0, new Float32Array([1 / nn, 0, 0, 0]));
    device.queue.writeBuffer(B.mixFirst, 0, new Float32Array([1 / nn, 0, 0, 0]));

    // ----- relax (damped wave) -----
    const rmod = await module(RELAX_WGSL(N), 'relax' + N);
    const rEntries = [[0, 'uniform'], [1, 'read-only-storage'], [2, 'read-only-storage'], [3, 'storage'], [4, 'storage'], [5, 'read-only-storage'], [6, 'read-only-storage'], [7, 'storage'], [8, 'storage'], [9, 'uniform']]
      .map(([binding, type]) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } }));
    const rbgl2 = device.createBindGroupLayout({ entries: rEntries });
    const rlay = device.createPipelineLayout({ bindGroupLayouts: [rbgl2] });
    const rp = {}; for (const e of ['step', 'compose', 'retarget', 'addKick', 'reduce']) rp[e] = device.createComputePipeline({ layout: rlay, compute: { module: rmod, entryPoint: e } });
    const E = [mk(nn * 4, ST | CS | CD), mk(nn * 4, ST | CS | CD)], Vb = [mk(nn * 4, ST | CS | CD), mk(nn * 4, ST | CS | CD)];
    const ambU = mk(192, UN | CD), hp = mk(nn * 4, ST | CS | CD), rstats = mk(16, ST | CS), rU = mk(32, UN | CD);
    const statStage = mk(16, GPUBufferUsage.MAP_READ | CD), stageB = mk(nn * 4, GPUBufferUsage.MAP_READ | CD), stageE = mk(nn * 4, GPUBufferUsage.MAP_READ | CD);
    const bgP = [0, 1].map(k => device.createBindGroup({ layout: rbgl2, entries: [
      { binding: 0, resource: { buffer: rU } }, { binding: 1, resource: { buffer: E[k] } }, { binding: 2, resource: { buffer: Vb[k] } },
      { binding: 3, resource: { buffer: E[1 - k] } }, { binding: 4, resource: { buffer: Vb[1 - k] } }, { binding: 5, resource: { buffer: B.field } },
      { binding: 6, resource: { buffer: hp } }, { binding: 7, resource: { buffer: B.hT } }, { binding: 8, resource: { buffer: rstats } }, { binding: 9, resource: { buffer: ambU } }] }));
    const ctx = { N, P, B, bg, layout, texA, texB, renderBG: null, cv: [document.createElement('canvas'), document.createElement('canvas')], raw: [document.createElement('canvas'), document.createElement('canvas')], rp, E, Vb, hp, ambU, kV: hp, rstats, rU, statStage, stageB, stageE, bgP, cur: 0 };
    // ----- god-ray light volume -----
    const gmod = await module(godWGSL(N), 'god' + N);
    const gbgl = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ...[1, 2, 3].map(b => ({ binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } })),
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '3d' } },
    ] });
    const glay = device.createPipelineLayout({ bindGroupLayouts: [gbgl] });
    ctx.god = { gp: {}, gU: mk(64, UN | CD), M: 0, NZ: 0, vol: null };
    for (const e of ['vclear', 'vsplat', 'vblurX', 'vblurY']) ctx.god.gp[e] = device.createComputePipeline({ layout: glay, compute: { module: gmod, entryPoint: e } });
    ctx.god.gbgl = gbgl;
    return ctxCache[N] = ctx;
  }
  // (re)allocate the volume for the current M / NZ settings and rebuild the render bind group
  function ensureVol(c) {
    const g = c.god, M = Math.min(c.N, +$('volres').value), NZ = Math.max(2, Math.round(rd('nz')));
    if (g.M === M && g.NZ === NZ) return;
    g.vol?.destroy(); g.accum?.destroy(); g.tmp?.destroy();
    g.M = M; g.NZ = NZ; g.sub = g.sub || 1;
    g.vol = device.createTexture({ size: [M, M, NZ], dimension: '3d', format: 'r32float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    g.accum = device.createBuffer({ size: M * M * NZ * 4, usage: ST | CS | CD }); g.tmp = device.createBuffer({ size: M * M * NZ * 4, usage: ST | CS | CD });
    g.bg = device.createBindGroup({ layout: g.gbgl, entries: [
      { binding: 0, resource: { buffer: g.gU } }, { binding: 1, resource: { buffer: c.B.hx } }, { binding: 2, resource: { buffer: c.B.hy } }, { binding: 3, resource: { buffer: c.B.hT } },
      { binding: 4, resource: { buffer: g.accum } }, { binding: 5, resource: { buffer: g.tmp } }, { binding: 6, resource: g.vol.createView({ dimension: '3d' }) }] });
    c.renderBG = device.createBindGroup({ layout: rbgl, entries: [
      { binding: 0, resource: { buffer: ubuf } }, { binding: 1, resource: { buffer: c.B.hT } }, { binding: 2, resource: { buffer: c.B.hx } },
      { binding: 3, resource: { buffer: c.B.hy } }, { binding: 4, resource: { buffer: c.B.C } },
      { binding: 5, resource: g.vol.createView({ dimension: '3d' }) }, { binding: 6, resource: vsamp }] });
  }
  function writeGod(c) {
    const g = c.god, ab = new ArrayBuffer(64), f = new Float32Array(ab), u = new Uint32Array(ab), sig = 2 * g.M / 512;
    f[0] = rd('D'); f[1] = 1 / 1.33; u[2] = g.M; u[3] = g.NZ; u[4] = g.sub; f[5] = c.N / g.M; f[6] = Math.max(sig, 0.3); u[7] = Math.ceil(3 * f[6]); f[8] = rd('rnoise'); f[9] = (performance.now() - S.t0) / 1000;
    const L = S.L || lightParams(90, 0, f[0]); f[10] = L.di[0]; f[11] = L.di[1]; f[12] = L.di[2]; f[13] = L.fadeW;
    device.queue.writeBuffer(g.gU, 0, ab);
  }
  function encodeVolume(pass, c) {
    const g = c.god, wx = g.M / 8, z = g.NZ; pass.setBindGroup(0, g.bg);
    for (const e of ['vclear', 'vsplat', 'vblurX', 'vblurY']) { pass.setPipeline(g.gp[e]); pass.dispatchWorkgroups(wx, wx, z); }
  }

  // ---------- state ----------
  const S = { ctx: null, N: 512, paused: true, t0: performance.now() };
  const R = { t: 0, tl: 0, idx: 0, playing: true, holdT: 0, resid: 1, e2: 0, h2: 1e-4, emax: 0, nan: 0, rmsStar: 0.01, busy: false, slowed: false, nSub: 0, hSub: 0, listStr: null, seed: 7 };
  const clk = { off: 0, key: null, last: null, tickCount: 0, ticks: 0, lastShim: 0, shimIdx: 0, lastTickAt: 0 };
  const clockOn = () => $('clockOn').checked;
  app.state = S;
  const clamp01 = x => Math.min(1, Math.max(0, x));
  const rd = id => +$(id).value;
  const letters = () => { const L = $('textList').value.split(/[,，]/).map(x => x.trim()).filter(Boolean); return L.length ? L : [' ']; };
  const curLetter = () => { if (clockOn()) { if (!clk.key) clk.key = clockKey(); return clk.key; } return letters()[R.idx % letters().length]; };
  // ---------- clock ----------
  const pad2 = n => String(n).padStart(2, '0');
  const shownStr = () => { const [H, M, Sx] = clockParts(); return H + M + (showSec() ? Sx : ''); };   // displayed digits, e.g. '0905'
  const clockDate = () => new Date(Date.now() + clk.off);
  const showSec = () => $('clockSec').checked || $('clockLayout').value === 'seconds';
  function clockParts() {
    const d = clockDate(); let H = d.getHours(); if ($('clock12').value === '12') H = H % 12 || 12;
    return [pad2(H), pad2(d.getMinutes()), pad2(d.getSeconds())];
  }
  // glyph key: CLK|layout|line1|line2|line3 (a layout of 'single' has one line). Doubles as the cache key of the raw glyph canvas.
  function clockKey() {
    const [H, M, Sx] = clockParts(), sec = showSec(), lay = $('clockLayout').value === 'single' ? 'single' : 'stacked';
    if (lay === 'single') return 'CLK|single|' + H + M + (sec ? Sx : '');
    return 'CLK|stacked|' + H + '|' + M + (sec ? '|' + Sx : '');
  }

  // ---------- dynamics parameters / stability ----------
  // seconds shown -> the surface is retargeted every second; stiffer/damper water (kappa x1.8, gamma x1.6) so it settles quickly
  const dynP = () => { const q = clockOn() && showSec() ? [1.8, 1.6] : [1, 1]; return dynP0(q); };
  const dynP0 = q => ({ c2: rd('c') ** 2, kappa: rd('kappa') * q[0], gamma: rd('gamma') * q[1], nu: rd('nu') * 1e-5, s: rd('s') * 1e-7, kmax: Math.min(rd('kmax'), S.N / 8), cap: rd('sub') });   // max-k clipped to the grid (N/8 cycles)
  // Symplectic Euler with explicit friction, per Fourier mode: stable iff dt^2 w^2 < 4 - 2 dt G  (w^2 = omega^2, G = gamma + nu*lambda, lambda = 8N^2 = 5-pt Laplacian max).
  // omega^2_max = c^2*8N^2 + s*64N^4 + kappa (5-/13-point stencils, h = 1/N). We take 80 % of the limit: w^2 dt^2 + 2 G dt = 3.2  ->  dt = (-G + sqrt(G^2 + 3.2 w^2)) / w^2.
  function dtSafe(P, N) {
    const w2 = P.c2 * 8 * N * N + P.s * 64 * N ** 4 + P.kappa, G = P.gamma + P.nu * 8 * N * N;
    return (-G + Math.sqrt(G * G + 3.2 * w2)) / w2;
  }
  function kappaEff(P, t) { const r = rd('ramp'); if (r <= 0) return P.kappa; const q = clamp01(t / r); return P.kappa * q * q * (3 - 2 * q); }
  function planSteps(dt, P, N) {
    const ds = dtSafe(P, N); let n = Math.max(1, Math.ceil(dt / ds)), slowed = false;
    if (n > P.cap) { n = P.cap; slowed = dt / n > ds; }
    return { n, h: Math.min(dt / n, ds), slowed, ds };
  }
  function writeDyn(c, P, h, t) { device.queue.writeBuffer(c.rU, 0, new Float32Array([h, P.c2, kappaEff(P, t), P.gamma, P.nu, P.s, c.N * c.N, 0])); }
  const d8 = (pass, c, pipe) => { pass.setPipeline(pipe); pass.dispatchWorkgroups(c.N / 8, c.N / 8); };
  const d1 = (pass, pipe) => { pass.setPipeline(pipe); pass.dispatchWorkgroups(1); };
  // ---------- continuous ambient forcing: e_tt += F(x,t), F = sum_m A_m mod_m(t) cos(2pi k_m.x - phi_m(t)) ----------
  // 12 integer wave vectors (periodic) around |k| ~ 'ambient scale' cycles; frequency Omega_m = speed * omega0(k_m) (dispersion relation of the model, speed<1 = off-resonant);
  // slow amplitude modulation (10-30 s) with random phases; phases advance with simulation time. The gain is normalised analytically with the steady-state transfer
  // |H| = 1/sqrt((w0^2-W^2)^2 + (G W)^2), G = gamma + nu k^2, so that rms(e) = ambient * rms(h*) whatever N / kappa / gamma / nu / c / s are.
  const AMB = { modes: [], scaleKey: null, w: new Float32Array(48), simT: 0, seed: 4242, MA: 0.6 };
  function ambModes(scale) {
    const rnd = mulberry32(AMB.seed), out = [], TP = 2 * Math.PI; let guard = 0;
    while (out.length < 12 && guard++ < 5000) {
      const kmin = Math.max(1.5, scale * 0.5), kmax = scale * 1.5 + 1;
      const kx = Math.round((rnd() * 2 - 1) * kmax), ky = Math.round((rnd() * 2 - 1) * kmax), kk = Math.hypot(kx, ky);
      if (kk < kmin || kk > kmax) continue;
      if (out.some(o => (o.kx === kx && o.ky === ky) || (o.kx === -kx && o.ky === -ky))) continue;
      out.push({ kx, ky, kk, ph: rnd() * TP, T: 10 + rnd() * 20, psi: rnd() * TP, dir: rnd() < 0.5 ? 1 : -1, w: 1 / Math.sqrt(kk), phi: rnd() * TP });
    }
    return out;
  }
  function ambUpdate(c, dt, P) {
    const on = $('ambOn').checked, r = rd('ambient'), sc = Math.round(rd('ascale') * 4) / 4, w = AMB.w; w.fill(0);
    if (AMB.scaleKey !== sc) { AMB.modes = ambModes(sc); AMB.scaleKey = sc; }
    if (!on || r <= 0 || !(R.rmsStar > 0)) { device.queue.writeBuffer(c.ambU, 0, w); return; }
    const TP = 2 * Math.PI, M = AMB.modes, sp = rd('aspeed'); AMB.simT += dt;
    let sw2 = 0; for (const m of M) sw2 += m.w * m.w;   // E_m = g w_m; sum E_m^2 <mod^2>/2 = target^2
    const g = r * R.rmsStar / Math.sqrt(sw2 * (1 + AMB.MA * AMB.MA / 2) / 2);
    M.forEach((m, i) => {
      const k = TP * m.kk, w0sq = P.c2 * k * k + P.kappa + P.s * k ** 4, w0 = Math.sqrt(w0sq), W = sp * w0, G = P.gamma + P.nu * k * k;
      const D = Math.sqrt((w0sq - W * W) ** 2 + (G * W) ** 2);   // forcing amplitude per unit e amplitude
      m.phi += m.dir * W * dt;
      const mod = 1 + AMB.MA * Math.sin(TP * AMB.simT / m.T + m.psi);
      w[4 * i] = TP * m.kx; w[4 * i + 1] = TP * m.ky; w[4 * i + 2] = m.phi + m.ph; w[4 * i + 3] = g * m.w * D * mod;
    });
    device.queue.writeBuffer(c.ambU, 0, w);
  }
  function encodeSteps(pass, c, n) { for (let i = 0; i < n; i++) { pass.setPipeline(c.rp.step); pass.setBindGroup(0, c.bgP[c.cur]); pass.dispatchWorkgroups(c.N / 8, c.N / 8); c.cur ^= 1; } }

  // ---------- solver (h*) ----------
  const PRE_FINAL = 1.5;
  const paramsFinal = () => ({ D: rd('D'), contrast: rd('contrast'), preblur: PRE_FINAL * S.N / 512, blur: 2 * rd('rblur') * S.N / 512, iters: rd('iters'), beta: 0.7, sub: rd('subray'), morph: 0, el: rd('elev'), az: rd('az'), center: $('center').checked, wrap: $('wrap').checked, cast: rd('cast'), fit: $('fit').checked });
  function writeParams(c, p) {
    const k = p.D * (1 - 1 / 1.33), nn = c.N * c.N;
    const L = lightParams(p.el, p.az, p.D, p.center, p.wrap); S.L = L;
    device.queue.writeBuffer(c.B.params, 0, new Float32Array([p.D, k, p.contrast, p.morph, p.preblur, p.blur, p.sub, 1.33, p.beta, 0, 0, 0,
      L.K[0], L.K[1], L.K[2], L.shDes[0], L.shDes[1], L.sOverD[0], L.sOverD[1], L.di[0], L.di[1], L.di[2], L.fadeW, 0]));
    device.queue.writeBuffer(c.B.mixIter, 0, new Float32Array([p.beta / nn, 1 - p.beta, 0, 0]));
  }
  function drawClockGlyph(c, key, N) {
    const parts = key.split('|'), lay = parts[1], lines = parts.slice(2), fam = '"Arial Black", "Helvetica Neue", Helvetica, sans-serif';
    const base = lay === 'single' ? [0.5] : (lines.length === 3 ? [0.46, 0.46, 0.26] : [0.62, 0.62]);   // font size / N before fitting
    const meas = (txt, fs) => { c.font = `900 ${fs}px ${fam}`; const m = c.measureText(txt); return { asc: c.measureText('0').actualBoundingBoxAscent, L: m.actualBoundingBoxLeft, R: m.actualBoundingBoxRight }; };
    const gap = 0.07 * N; let s = 1, tot = 0;
    const calc = () => { tot = 0; let wmax = 0; lines.forEach((t, i) => { const m = meas(t, base[i] * N * s); tot += m.asc; wmax = Math.max(wmax, m.L + m.R); }); tot += gap * (lines.length - 1); return { wmax }; };
    for (let it = 0; it < 3; it++) { const { wmax } = calc(); const f = Math.min(1, 0.86 * N / tot, 0.88 * N / Math.max(wmax, 1)); if (f >= 0.999) break; s *= f; }
    calc(); let y = (N - tot) / 2;
    lines.forEach((t, i) => {
      const fs = base[i] * N * s, m = meas(t, fs); c.font = `900 ${fs}px ${fam}`; c.textAlign = 'left'; c.textBaseline = 'alphabetic';
      const x = N / 2 - (m.R - m.L) / 2; c.lineJoin = 'round'; c.lineWidth = fs * 0.035; c.strokeStyle = '#fff';   // slight embolden so '1' / thin strokes survive blur
      c.fillText(t, x, y + m.asc); c.strokeText(t, x, y + m.asc); y += m.asc + gap;
    });
  }
  function drawGlyph(cv, text, N) {
    const c = cv.getContext('2d', { willReadFrequently: true }); cv.width = cv.height = N;
    c.fillStyle = '#000'; c.fillRect(0, 0, N, N); c.fillStyle = '#fff';
    if (text.startsWith('CLK|')) { drawClockGlyph(c, text, N); return; }
    let fs = N * 0.85; const fam = '"Arial Black", Helvetica, sans-serif';
    c.font = `900 ${fs}px ${fam}`;
    const wdt = c.measureText(text).width; if (wdt > N * 0.92) { fs *= N * 0.92 / wdt; c.font = `900 ${fs}px ${fam}`; }
    c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillText(text, N / 2, N * 0.54);
  }
  // Cast-shadow warp of the glyph (floor plane, about the floor centre): stretch sa = 1 + cast*(1/sin(el) - 1) along the azimuth ('sin' law = beam-footprint stretch;
  // app.stretchLaw='tan' = upright-object shadow length), 1 perpendicular; optional uniform shrink so the warped glyph bbox fits inside ~0.8 of the floor (scale-to-fit).
  app.stretchLaw = 'sin';
  function castWarp(c, q) {
    const N = c.N, cast = q.cast || 0; c.warpInfo = { sa: 1, f: 1 };
    for (const cv of c.cv) cv.width = cv.height = N;
    if (cast <= 0) { c.cv[0].getContext('2d').drawImage(c.raw[0], 0, 0); return; }
    const el = Math.min(90, Math.max(30, q.el)), a = q.az * rad;
    const sa = Math.min(3, Math.max(0.3, 1 + cast * ((app.stretchLaw === 'tan' ? 1 / Math.tan(el * rad) : 1 / Math.sin(el * rad)) - 1)));
    if (!c.bbox || c.bboxKey !== c.rawKey) {
      let x0 = N, x1 = 0, y0 = N, y1 = 0; const d = c.raw[0].getContext('2d', { willReadFrequently: true }).getImageData(0, 0, N, N).data;
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) if (d[4 * (y * N + x)] > 16) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
      c.bbox = [x0, y0, x1, y1]; c.bboxKey = c.rawKey;
    }
    const [x0, y0, x1, y1] = c.bbox, h = N / 2, ca = Math.cos(a), sn = Math.sin(a);
    const M = (px, py) => { const u = px * ca + py * sn, v = -px * sn + py * ca; const u2 = u * sa; return [u2 * ca - v * sn, u2 * sn + v * ca]; };
    let mx = 0, my = 0, ox = 0;
    for (const [px, py] of [[x0, y0], [x1 + 1, y0], [x0, y1 + 1], [x1 + 1, y1 + 1]]) { const r = M(px - h, py - h); mx = Math.max(mx, Math.abs(r[0])); my = Math.max(my, Math.abs(r[1])); ox = Math.max(ox, Math.abs(px - h), Math.abs(py - h)); }
    let f = 1;
    if (q.fit) { const t = Math.min(1, cast / 0.2), lim = 0.4 * N + (1 - t) * Math.max(0, ox - 0.4 * N); f = Math.min(1, lim / Math.max(mx, my, 1e-6)); }
    c.warpInfo = { sa, f, bbox: c.bbox };
    const g = c.cv[0].getContext('2d'); g.fillStyle = '#000'; g.fillRect(0, 0, N, N); g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
    g.save(); g.translate(h, h); g.scale(f, f); g.rotate(a); g.scale(sa, 1); g.rotate(-a); g.translate(-h, -h); g.drawImage(c.raw[0], 0, 0); g.restore();
  }
  function uploadGlyph(c = S.ctx, q = paramsFinal(), text = curLetter()) {
    const N = c.N; if (c.rawKey !== text + N) { drawGlyph(c.raw[0], text, N); c.rawKey = text + N; }
    castWarp(c, q); device.queue.copyExternalImageToTexture({ source: c.cv[0] }, { texture: c.texA }, [N, N]);
  }
  function fft3(pass, c, colsPipe) { pass.setPipeline(c.P.rowsFwd); pass.dispatchWorkgroups(c.N); pass.setPipeline(colsPipe); pass.dispatchWorkgroups(c.N); pass.setPipeline(c.P.rowsInv); pass.dispatchWorkgroups(c.N); }
  function encodeSolve(pass, c, p) {
    pass.setBindGroup(0, c.bg.pre); d8(pass, c, c.P.mixTex); fft3(pass, c, c.P.colsPre);
    d1(pass, c.P.red0); d8(pass, c, c.P.prepT); d1(pass, c.P.red1); d8(pass, c, c.P.normT);
    pass.setBindGroup(0, c.bg.first); d8(pass, c, c.P.initT); fft3(pass, c, c.P.colsPoisson);
    pass.setBindGroup(0, c.bg.iter);
    for (let i = 0; i < p.iters; i++) { d8(pass, c, c.P.maWarp); fft3(pass, c, c.P.colsPoisson); }
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
  const godOn = () => $('god').checked;
  // forward pass: hT = h* + e ; grad ; splat ; blur -> C
  function encodeForward(pass, c) {
    pass.setPipeline(c.rp.compose); pass.setBindGroup(0, c.bgP[c.cur]); pass.dispatchWorkgroups(c.N / 8, c.N / 8);
    pass.setBindGroup(0, c.bg.R); d8(pass, c, c.P.grad); d8(pass, c, c.P.clearAcc); d8(pass, c, c.P.splat);
    pass.setBindGroup(0, c.bg.c); d8(pass, c, c.P.accumToC); fft3(pass, c, c.P.colsC);
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
  async function readBuf(src, stage) { const c = S.ctx, n = c.N * c.N, enc = device.createCommandEncoder(); enc.copyBufferToBuffer(src, 0, stage, 0, n * 4); device.queue.submit([enc.finish()]);
    await stage.mapAsync(GPUMapMode.READ); const a = new Float32Array(stage.getMappedRange().slice(0)); stage.unmap(); return a; }

  // ---------- initial condition / disturbance ----------
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
    device.queue.writeBuffer(c.kV, 0, ic.v);
    const enc = device.createCommandEncoder(), ps = enc.beginComputePass(); ps.setPipeline(c.rp.addKick); ps.setBindGroup(0, c.bgP[1 - c.cur]); ps.dispatchWorkgroups(c.N / 8, c.N / 8); ps.end();
    device.queue.submit([enc.finish()]); R.holdT = 0; R.tl = 0;
  }
  // ----- clock water stirring: velocity-only ring from the centre (tick) + faint random shimmer, all through the same addKick path -----
  function kickBuf(c) {
    if (c.kk) return c.kk; const N = c.N, ring = new Float32Array(N * N); let m = 0;
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) { const dx = (i + 0.5) / N - 0.5, dy = (j + 0.5) / N - 0.5, r = Math.hypot(dx, dy), q = (r - 0.13) / 0.08;
      const v = Math.exp(-q * q) * Math.cos(2 * Math.PI * (r - 0.13) / 0.14); ring[j * N + i] = v; m += v; }
    m /= N * N; let s = 0; for (let i = 0; i < ring.length; i++) { ring[i] -= m; s += ring[i] * ring[i]; } s = Math.sqrt(s / ring.length) || 1; for (let i = 0; i < ring.length; i++) ring[i] /= s;   // unit rms
    return c.kk = { ring, out: new Float32Array(N * N) };
  }
  function applyKick(c, src, amp) {
    const kb = kickBuf(c), o = kb.out; for (let i = 0; i < o.length; i++) o[i] = src[i] * amp;
    device.queue.writeBuffer(c.kV, 0, o);
    const enc = device.createCommandEncoder(), ps = enc.beginComputePass(); ps.setPipeline(c.rp.addKick); ps.setBindGroup(0, c.bgP[1 - c.cur]); ps.dispatchWorkgroups(c.N / 8, c.N / 8); ps.end(); device.queue.submit([enc.finish()]);
  }
  function clockTick() {
    const c = S.ctx; R.tl = 0; R.resid = 1; R.holdT = 0; clk.ticks++; clk.lastTickAt = performance.now(); solveStar(true);
    const k = rd('kick') * (showSec() ? 0.35 : 1); if (k > 0) applyKick(c, kickBuf(c).ring, k * 4 * R.rmsStar);
  }
  function nextLetter(to) {
    if (letters().length <= 1) { disturb(); return; }
    R.idx = to != null ? to : (R.idx + 1) % letters().length; R.tl = 0; R.holdT = 0; R.resid = 1; solveStar(true);
  }

  const meshM = c => $('decouple').checked ? Math.min(c.N, Math.max(2, +$('mesh').value || c.N)) : c.N;
  // ---------- render ----------
  function encodeRender(enc, c) {
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: msaaTex.createView(), resolveTarget: gctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'discard', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
      depthStencilAttachment: { view: depthTex.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' },
    });
    rp.setBindGroup(0, c.renderBG);
    rp.setPipeline(RP.bg); rp.draw(3); rp.setPipeline(RP.table); rp.draw(6); rp.setPipeline(RP.floor); rp.draw(6);
    rp.setPipeline(RP.water); rp.draw((meshM(c) - 1) * (meshM(c) - 1) * 6); rp.setPipeline(RP.glass); rp.draw(6, 4); rp.end();
  }
  // ---------- camera ----------
  const PRESETS = { 1: { yaw: 0, pitch: 89.5, dist: 3.4 }, 2: { yaw: 22, pitch: 52, dist: 3.3 }, 3: { yaw: 8, pitch: 14, dist: 3.1 } };
  const cam = { yaw: 22, pitch: 21, dist: 3.3, ty: 22, tp: 21, td: 3.3, vy: 0, vp: 0, oy: 0, toy: 0 };   // oy = look-at height offset (world units) added to the default target; toy = eased target. x/z of the target are fixed at 0
  const rad = Math.PI / 180; app.cam = cam;
  function preset(k) { const p = PRESETS[k]; // take the shortest way round in yaw
    cam.ty = p.yaw; cam.tp = p.pitch; cam.td = p.dist; cam.yaw = cam.ty + ((((cam.yaw - cam.ty) % 360) + 540) % 360 - 180); }
  app.preset = k => { preset(k); cam.vy = cam.vp = 0; };   // presets no longer stop the auto-orbit
  app.snapCam = () => { cam.yaw = cam.ty; cam.pitch = cam.tp; cam.dist = cam.td; cam.oy = cam.toy; };
  const syncTargetSliders = () => { $('tgy').value = (+cam.toy).toFixed(2); };
  const readTargetSliders = () => { cam.toy = +$('tgy').value; };
  $('tgy').addEventListener('input', readTargetSliders);
  $('tgreset').onclick = () => { cam.toy = 0; syncTargetSliders(); $('tgy').dispatchEvent(new Event('change', { bubbles: true })); };
  let drag = null, moved = 0, dragEnd = -1e9;
  canvas.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, t: performance.now(), pan: e.button === 2 || e.shiftKey }; moved = 0; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointermove', e => {
    if (!drag) return; if (e.buttons === 0) { drag = null; dragEnd = performance.now(); return; } const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY; moved += Math.abs(dx) + Math.abs(dy);
    if (drag.pan) {   // pan: vertical drag moves only the look-at height, scaled by distance
      const sc = 2 * cam.dist * Math.tan(FOV / 2) / Math.max(1, canvas.clientHeight);
      cam.toy = Math.min(1, Math.max(-1, cam.toy + dy * sc)); syncTargetSliders();
    } else { cam.ty -= dx * 0.3; cam.tp = Math.min(89.5, Math.max(4, cam.tp + dy * 0.3)); }
  });
  canvas.addEventListener('pointerup', e => { drag = null; dragEnd = performance.now(); });
  canvas.addEventListener('contextmenu', e => e.preventDefault());
  canvas.addEventListener('wheel', e => { e.preventDefault(); cam.td = Math.min(6, Math.max(1.2, cam.td * Math.exp(e.deltaY * 0.001))); }, { passive: false });
  const FOV = 34 * rad;

  // ---------- frame ----------
  function writeUniforms() {
    const D = rd('D'), tgt = [0, -D * 0.42 + cam.oy, 0];
    const cp = Math.cos(cam.pitch * rad), eye = [tgt[0] + cam.dist * cp * Math.sin(cam.yaw * rad), tgt[1] + cam.dist * Math.sin(cam.pitch * rad), tgt[2] + cam.dist * cp * Math.cos(cam.yaw * rad)];
    const up = cam.pitch > 89 ? [0, 0, -1] : [0, 1, 0];
    const lk = lookAt(eye, tgt, up), asp = rw / rh;
    const vp = mul(perspective(FOV, asp, 0.05, 60), lk.m);
    const u = new Float32Array(64);
    u.set(vp, 0);
    u.set([eye[0], eye[1], eye[2], (performance.now() - S.t0) / 1000], 16);
    u.set([-lk.z[0], -lk.z[1], -lk.z[2], Math.tan(FOV / 2)], 20);
    u.set([lk.x[0], lk.x[1], lk.x[2], asp], 24);
    u.set([lk.y[0], lk.y[1], lk.y[2], 0], 28);
    u.set([D, rd('relief'), rd('expo'), S.N], 32);
    const gl = rd('glow');
    u.set([1.0 + 0.8 * gl, $('glass').checked ? 1 : 0, rw, rh], 36);
    u.set([gl, 0, 0, 0], 40);
    u.set([godOn() ? 1 : 0, rd('rgain'), rd('rsig'), rd('rg')], 44);
    u.set([rd('rsteps'), rd('rnoise'), S.ctx.god.NZ || 24, 0], 48);
    const L = S.L || lightParams(90, 0, D), sw = L.sunWorld, ce = Math.cos(L.el * rad), eta = 1 / NW;
    u.set([sw[0], sw[1], sw[2], 0], 52);
    const wy = -Math.sqrt(1 - eta * eta * ce * ce), wl = Math.hypot(eta * ce, wy);   // refracted travel direction at a flat surface (world x, up, z); world z == solver y
    u.set([eta * L.di[0] / wl, wy / wl, eta * L.di[1] / wl, 0], 56);
    u.set([meshM(S.ctx), $('flat').checked ? 1 : 0, $('csamp').value === 'nearest' ? 1 : 0, 0], 60);
    device.queue.writeBuffer(ubuf, 0, u.subarray(0, 64));
  }
  const hudEl = $('hud'), camEl = $('camv');
  function updateHud() {
    hudEl.style.display = $('hudOn').checked ? 'block' : 'none'; if (!$('hudOn').checked) return;
    const r = R.resid, bar = '#'.repeat(Math.max(0, Math.min(30, Math.round(30 * Math.min(1, r))))).padEnd(30, '.');
    const st = R.holdT > 0 ? `settled, holding ${R.holdT.toFixed(1)} / ${rd('hold')} s` : 'relaxing';
    hudEl.textContent = `${clockOn() ? 'clock ' + shownStr() + ' (' + clk.ticks + ' ticks)' : 'letter ' + curLetter()}   t = ${R.tl.toFixed(1)} s   residual |e|/|h*| = ${r.toFixed(3)}  [${bar}]  ${st}` + (R.slowed ? '   (sim slowed: substep cap)' : '');
  }
  let tPrev = performance.now(), frameCount = 0, tFps = performance.now(), fr = 0;
  const LBL = ['c', 'kappa', 'gamma', 'nu', 's', 'sub', 'amp', 'kmax', 'seed', 'hold', 'thr', 'ramp', 'glow', 'D', 'contrast', 'iters', 'expo', 'relief', 'rgain', 'rsig', 'rsteps', 'rg', 'rnoise', 'nz', 'elev', 'az', 'cast', 'rblur', 'subray', 'kick', 'ascale', 'aspeed', 'ambient', 'ospeed', 'tgy'];
  function frame(now) {
    requestAnimationFrame(frame);
    const c = S.ctx; if (!c || S.paused) return;
    ensureVol(c);
    const dtR = Math.min(0.05, (now - tPrev) / 1000); tPrev = now; fr++;
    for (const k of LBL) $(k + 'v').textContent = $(k).value;
    if ($('orbit').checked && !drag && now - dragEnd > 1500) cam.ty += rd('ospeed') * dtR;   // orbit pauses during a drag, resumes 1.5 s after release   // deg/s, dtR in s
    const a = 1 - Math.exp(-dtR * 9);
    cam.yaw += (cam.ty - cam.yaw) * a; cam.pitch += (cam.tp - cam.pitch) * a; cam.dist += (cam.td - cam.dist) * a;
    cam.oy += (cam.toy - cam.oy) * a;
    camEl.textContent = `camera  yaw ${(((cam.yaw % 360) + 360) % 360).toFixed(1)}  pitch ${cam.pitch.toFixed(1)}  dist ${cam.dist.toFixed(2)}  target y ${cam.oy.toFixed(2)}`;
    fitCanvas();
    if (clockOn()) { R.listStr = $('textList').value; if (!R.busy) { clk.key = clockKey(); if (clk.key !== clk.last) clockTick(); } }
    else if (R.listStr !== $('textList').value && !R.busy) { R.listStr = $('textList').value; nextLetter(0); }
    const enc = device.createCommandEncoder(), ps = enc.beginComputePass();
    let simDt = 0;
    if (R.playing && !R.busy) {
      const P = dynP(), pl = planSteps(dtR, P, c.N); R.slowed = pl.slowed; R.nSub = pl.n; R.hSub = pl.h;
      writeDyn(c, P, pl.h, R.t); simDt = pl.n * pl.h; ambUpdate(c, simDt, P); encodeSteps(ps, c, pl.n);
    }
    encodeForward(ps, c); ps.end();
    writeUniforms(); encodeRender(enc, c); device.queue.submit([enc.finish()]);
    if (R.playing && !R.busy) {
      R.t += simDt; R.tl += simDt;
      if (R.resid < rd('thr') && R.tl > 0.5) R.holdT += dtR; else R.holdT = 0;
      if (!clockOn() && (R.holdT >= rd('hold') || R.tl > 60)) nextLetter();
    }
    if ((!isFinite(R.resid) || R.nan > 0 || R.resid > 50) && !R.busy) { console.warn('relax: blow-up detected, re-initialising'); initSim().then(() => {}); }
    if (fr % 6 === 0 && !R.busy && !R.statBusy) { R.statBusy = true; readStats().catch(() => {}).finally(() => { R.statBusy = false; }); }
    updateHud();
    app.frames++; frameCount++;
    if (now - tFps > 500) { app.fps = frameCount * 1000 / (now - tFps); frameCount = 0; tFps = now; const mm = meshM(c); $('fps').textContent = `${app.fps.toFixed(0)} fps  N=${S.N}  ${rw}x${rh}\nmesh ${mm}x${mm}  triangles: ${(2 * (mm - 1) ** 2).toLocaleString('en-US')}\nsubsteps ${R.nSub} x ${(R.hSub * 1000).toFixed(2)} ms` + (app.lastError ? '\nERROR ' + app.lastError : ''); }
  }

  // wave resolution: N = surface vertices = rays traced = caustic resolution. On change the new h* is solved and e is resampled from the old surface (h_old - h*_new): no pop.
  const resamp = (src, No, Nn) => { const o = new Float32Array(Nn * Nn), k = No / Nn;
    for (let y = 0; y < Nn; y++) { const fy = y * k, y0 = Math.floor(fy), ay = fy - y0, ya = y0 % No, yb = (y0 + 1) % No;
      for (let x = 0; x < Nn; x++) { const fx = x * k, x0 = Math.floor(fx), ax = fx - x0, xa = x0 % No, xb = (x0 + 1) % No;
        o[y * Nn + x] = (src[ya * No + xa] * (1 - ax) + src[ya * No + xb] * ax) * (1 - ay) + (src[yb * No + xa] * (1 - ax) + src[yb * No + xb] * ax) * ay; } } return o; };
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
      R.holdT = 0; await readStats(); R.busy = false;
    }
    R.busy = false; S.paused = false;
  }

  // ---------- debug hooks ----------
  const corr = (a, b) => { const n = a.length; let ma = 0, mb = 0; for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; } ma /= n; mb /= n; let sab = 0, saa = 0, sbb = 0; for (let i = 0; i < n; i++) { const x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; } return sab / Math.sqrt(saa * sbb); };
  const stat = a => { let m = 0, s2 = 0, mx = -1e30, mn = 1e30; for (let i = 0; i < a.length; i++) { m += a[i]; s2 += a[i] * a[i]; mx = Math.max(mx, a[i]); mn = Math.min(mn, a[i]); } m /= a.length; return { mean: m, std: Math.sqrt(Math.max(0, s2 / a.length - m * m)), max: mx, min: mn }; };
  const frameOnce = async (render = true) => { const c = S.ctx; fitCanvas(); const enc = device.createCommandEncoder(), ps = enc.beginComputePass(); encodeForward(ps, c); ps.end(); if (render) { writeUniforms(); encodeRender(enc, c); } device.queue.submit([enc.finish()]); await device.queue.onSubmittedWorkDone(); };
  // ---------- GPU vs CPU / regression helpers ----------
  const grayOf = (cv, N) => { const d = cv.getContext('2d').getImageData(0, 0, N, N).data, g = new Float64Array(N * N); for (let i = 0; i < N * N; i++) g[i] = (0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2]) / 255; return g; };
  const relRms = (a, b) => { let ma = 0, mb = 0; const n = a.length; for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; } ma /= n; mb /= n; let e = 0, v = 0; for (let i = 0; i < n; i++) { e += ((a[i] - ma) - (b[i] - mb)) ** 2; v += (b[i] - mb) ** 2; } return Math.sqrt(e / (v + 1e-30)); };
  async function rb(buf, size) { const st = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | CD }); const e = device.createCommandEncoder(); e.copyBufferToBuffer(buf, 0, st, 0, size); device.queue.submit([e.finish()]);
    await st.mapAsync(GPUMapMode.READ); const o = new Float32Array(st.getMappedRange().slice(0)); st.unmap(); st.destroy(); return o; }
  // design h* then the forward caustic of h* alone (e = 0); P = alternative pipeline set (old solver)
  async function eqRun(c, p, P = c.P) {
    writeParams(c, p); const cc = { N: c.N, bg: c.bg, P }, enc = device.createCommandEncoder(), ps = enc.beginComputePass();
    encodeSolve(ps, cc, p); ps.end(); enc.copyBufferToBuffer(c.B.field, 0, c.B.hT, 0, c.N * c.N * 4);
    const p2 = enc.beginComputePass(); p2.setBindGroup(0, c.bg.R); d8(p2, cc, P.grad); d8(p2, cc, P.clearAcc); d8(p2, cc, P.splat); p2.setBindGroup(0, c.bg.c); d8(p2, cc, P.accumToC); fft3(p2, cc, P.colsC); p2.end();
    device.queue.submit([enc.finish()]); const nn = c.N * c.N * 4; return { h: await rb(c.B.field, nn), C: await rb(c.B.C, nn) };
  }
  const oldCache = {};
  async function oldPipes(c) {
    if (oldCache[c.N]) return oldCache[c.N]; const { computeWGSL: oldWGSL } = await import('./solver_old.js?v=' + V);
    const mod = await module(oldWGSL(c.N), 'solverOld' + c.N), P = {}; const pp = (entryPoint, constants) => device.createComputePipeline({ layout: c.layout, compute: { module: mod, entryPoint, constants } });
    for (const e of ['rowsFwd', 'rowsInv', 'mixTex', 'prepT', 'normT', 'initT', 'maWarp', 'grad', 'clearAcc', 'accumToC', 'accumCopy', 'splat']) P[e] = pp(e);
    P.colsPoisson = pp('colsMul', { MODE: 0 }); P.colsPre = pp('colsMul', { MODE: 1 }); P.colsC = pp('colsMul', { MODE: 2 });
    P.red0 = pp('reduce', { MODE: 0 }); P.red1 = pp('reduce', { MODE: 1 }); return oldCache[c.N] = P;
  }
  const rx = window.relax = {
    state: () => ({ t: R.t, tl: R.tl, idx: R.idx, letter: curLetter(), resid: R.resid, rmsStar: R.rmsStar, emax: R.emax, nan: R.nan, playing: R.playing, holdT: R.holdT, nSub: R.nSub, hSub: R.hSub, slowed: R.slowed, dtSafe: dtSafe(dynP(), S.N), N: S.N }),
    pause: (v = true) => { R.playing = !v; $('play').textContent = R.playing ? 'Pause' : 'Play'; },
    residual: async () => { const s = await readStats(); return s; },
    // deterministic stepping with fixed substep dt (no auto letter switching); T seconds of simulated time
    step: async (T, dt) => {
      const c = S.ctx, P = dynP(); dt = dt ?? dtSafe(P, c.N) * 0.98; R.playing = false; let n = Math.round(T / dt), tt = T;
      while (n > 0) { const m = Math.min(n, 48); writeDyn(c, P, dt, R.t); const enc = device.createCommandEncoder(), ps = enc.beginComputePass(); encodeSteps(ps, c, m); ps.end(); device.queue.submit([enc.finish()]); R.t += m * dt; R.tl += m * dt; n -= m; }
      await device.queue.onSubmittedWorkDone(); return R.t;
    },
    frame: frameOnce,
    getC: async () => { await frameOnce(false); return readBuf(S.ctx.B.C, S.ctx.stageB); },
    getH: async () => { await frameOnce(false); return readBuf(S.ctx.B.hT, S.ctx.stageB); },
    getE: async () => { const c = S.ctx; return readBuf(c.E[c.cur], c.stageE); },
    refC: async () => { const c = S.ctx, enc = device.createCommandEncoder(); enc.copyBufferToBuffer(c.B.field, 0, c.B.hT, 0, c.N * c.N * 4);   // caustic of h* alone (e = 0)
      const ps = enc.beginComputePass(); ps.setBindGroup(0, c.bg.R); d8(ps, c, c.P.grad); d8(ps, c, c.P.clearAcc); d8(ps, c, c.P.splat);
      ps.setBindGroup(0, c.bg.c); d8(ps, c, c.P.accumToC); fft3(ps, c, c.P.colsC); ps.end(); device.queue.submit([enc.finish()]);
      return readBuf(c.B.C, c.stageB); },
    corr, stat,
    metrics: async ref => { const C = await rx.getC(), s = stat(C), r = stat(ref); return { corr: corr(C, ref), cv: s.std / s.mean, cvRatio: (s.std / s.mean) / (r.std / r.mean), res: (await readStats()).resid }; },
    // sample metrics at listed times (s since the last init/retarget) by stepping with fixed dt
    curve: async (times, dt) => { const ref = await rx.refC(), out = []; let t0 = R.tl;
      for (const T of times) { await rx.step(T - (R.tl - t0), dt); const m = await rx.metrics(ref); out.push({ t: +(R.tl - t0).toFixed(2), ...m }); } return out; },
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
      const g = c.god, n = g.M * g.M * g.NZ, st = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | CD }), enc = device.createCommandEncoder();
      enc.copyTextureToBuffer({ texture: g.vol }, { buffer: st, bytesPerRow: g.M * 4, rowsPerImage: g.M }, [g.M, g.M, g.NZ]); device.queue.submit([enc.finish()]);
      await st.mapAsync(GPUMapMode.READ); const a = new Float32Array(st.getMappedRange().slice(0)); st.unmap(); st.destroy();
      const sl = [], mm = g.M * g.M; let nan = 0;
      for (let z = 0; z < g.NZ; z++) { const q = a.subarray(z * mm, (z + 1) * mm); for (let i = 0; i < mm; i++) if (!isFinite(q[i])) nan++; const t = stat(q); sl.push({ z: +(rd('D') * z / (g.NZ - 1)).toFixed(3), mean: +t.mean.toFixed(4), std: +t.std.toFixed(3), max: +t.max.toFixed(2), min: +t.min.toFixed(3) }); }
      return { M: g.M, NZ: g.NZ, bytes: n * 4, nan, slices: sl, data: a };
    },
    checkSliceD: async () => {
      const c = S.ctx, was = $('god').checked; $('god').checked = true; await frameOnce(false); $('god').checked = was;
      const g = c.god, M = g.M, S_ = c.N / M, st = device.createBuffer({ size: M * M * 4, usage: GPUBufferUsage.MAP_READ | CD }), enc = device.createCommandEncoder();
      enc.copyTextureToBuffer({ texture: g.vol, origin: [0, 0, g.NZ - 1] }, { buffer: st, bytesPerRow: M * 4, rowsPerImage: M }, [M, M, 1]);
      device.queue.submit([enc.finish()]); await st.mapAsync(GPUMapMode.READ); const sl = new Float32Array(st.getMappedRange().slice(0)); st.unmap(); st.destroy();
      const C = await readBuf(c.B.C, c.stageB), pt = new Float32Array(M * M), bl = new Float32Array(M * M);
      for (let y = 0; y < M; y++) for (let x = 0; x < M; x++) { pt[y * M + x] = C[y * S_ * c.N + x * S_]; let a = 0; for (let j = 0; j < S_; j++) for (let i = 0; i < S_; i++) a += C[(y * S_ + j) * c.N + x * S_ + i]; bl[y * M + x] = a / (S_ * S_); }
      return { corrPoint: corr(sl, pt), corrBlock: corr(sl, bl), sliceStat: stat(sl), floorStat: stat(pt) };
    },
    benchParts: async (K = 60) => {
      const c = S.ctx, P = dynP(), pl = planSteps(1 / 60, P, c.N), was = $('god').checked; fitCanvas(); ensureVol(c); writeUniforms(); writeDyn(c, P, pl.h, 0);
      const t = async (fn, n) => { for (let i = 0; i < 3; i++) fn(); await device.queue.onSubmittedWorkDone(); const t0 = performance.now(); for (let i = 0; i < n; i++) fn(); await device.queue.onSubmittedWorkDone(); return (performance.now() - t0) / n; };
      const full = () => { const enc = device.createCommandEncoder(), ps = enc.beginComputePass(); encodeSteps(ps, c, pl.n); encodeForward(ps, c); ps.end(); writeUniforms(); encodeRender(enc, c); device.queue.submit([enc.finish()]); };
      const out = { N: c.N, M: c.god.M, NZ: c.god.NZ, w: rw, h: rh, steps: rd('rsteps') };
      $('god').checked = false; out.frameOff = await t(full, K);
      $('god').checked = true; out.frameOn = await t(full, K);
      writeGod(c); out.volumeBuild = await t(() => { const enc = device.createCommandEncoder(), ps = enc.beginComputePass(); encodeVolume(ps, c); ps.end(); device.queue.submit([enc.finish()]); }, K);
      out.raymarchApprox = out.frameOn - out.frameOff - out.volumeBuild; $('god').checked = was; return out;
    },
    resolve: () => solveStar(true),
    getT: async () => rb(S.ctx.B.T, S.ctx.N * S.ctx.N * 4), getStar: async () => rb(S.ctx.B.field, S.ctx.N * S.ctx.N * 4),
    // ----- oblique tests: GPU vs CPU float64 reference, regression vs the vertical solver (solver_old.js = tank3d_relax/solver.js verbatim) -----
    setLight: async (o) => { rx.set(o); await new Promise(r => setTimeout(r, 400)); await device.queue.onSubmittedWorkDone(); },
    compareOblique: async (opts = {}) => {
      const N = opts.N || 128, pairs = opts.pairs || [[90, 0, true], [65, 30, true], [60, 45, true], [45, 135, true], [35, 200, true], [65, 30, false], [45, 135, false]];
      const was = R.playing; R.playing = false; await device.queue.onSubmittedWorkDone(); const rows = [];
      const c = await makeCtx(N), keep = S.paused; S.paused = true;
      try { for (const [el, az, center] of pairs) {
        const p = { ...paramsFinal(), D: opts.D || 0.6, el, az, center, wrap: !!opts.wrap, cast: opts.cast ?? 0.5, fit: true };
        uploadGlyph(c, p, opts.letter || 'A'); const gA = grayOf(c.cv[0], N);
        const g = await eqRun(c, p), L = lightParams(el, az, p.D, center, p.wrap), cpu = cpuMod.cpuRun(gA, gA, N, p, L);
        rows.push({ el, az, center, hErr: relRms(g.h, cpu.h), CErr: relRms(g.C, cpu.C), CCorr: +corr(g.C, cpu.C).toFixed(4), s: L.s.map(v => +v.toFixed(3)) });
      } } finally { S.paused = keep; R.playing = was; }
      return rows;
    },
    regression90: async (N = 256) => {
      const was = R.playing; R.playing = false; await device.queue.onSubmittedWorkDone(); const c = await makeCtx(N), keep = S.paused; S.paused = true;
      try { const p = { ...paramsFinal(), D: 1, el: 90, az: 0, center: true, wrap: false, cast: 0 };
        uploadGlyph(c, p, 'A'); const nw = await eqRun(c, p), od = await eqRun(c, p, await oldPipes(c));
        return { N, hRelRms: relRms(nw.h, od.h), CRelRms: relRms(nw.C, od.C), CCorr: corr(nw.C, od.C), Cmean: stat(nw.C).mean };
      } finally { S.paused = keep; R.playing = was; }
    },
    bench: async (K = 60, frameDt = 1 / 60) => {
      const c = S.ctx, P = dynP(), pl = planSteps(frameDt, P, c.N); fitCanvas(); writeUniforms(); writeDyn(c, P, pl.h, 0);
      const run = async n => { const t0 = performance.now(); for (let i = 0; i < n; i++) { const enc = device.createCommandEncoder(), ps = enc.beginComputePass(); encodeSteps(ps, c, pl.n); encodeForward(ps, c); ps.end(); encodeRender(enc, c); device.queue.submit([enc.finish()]); } await device.queue.onSubmittedWorkDone(); return (performance.now() - t0) / n; };
      await run(5); const ms = await run(K); return { N: c.N, substeps: pl.n, subDtMs: pl.h * 1000, w: rw, h: rh, msPerFrame: ms, fps: 1000 / ms };
    },
  };

  // ---------- UI ----------
  let solveTimer = 0, lastSolve = 0;
  // throttled re-solve (leading edge after >=120 ms since the last one, trailing edge guaranteed): live slider dragging keeps retargeting e smoothly
  const resolveSoon = () => { clearTimeout(solveTimer); const run = () => { if (R.busy) { solveTimer = setTimeout(run, 40); return; } lastSolve = performance.now(); solveStar(true); };
    solveTimer = setTimeout(run, Math.max(0, 120 - (performance.now() - lastSolve))); };
  for (const id of ['D', 'contrast', 'iters', 'rblur', 'subray', 'elev', 'az', 'center', 'wrap', 'cast', 'fit']) $(id).addEventListener('input', resolveSoon);
  const cmp = $('compass').getContext('2d');
  function drawCompass() {
    const el = rd('elev'), az = rd('az') * rad, r = 14, len = r * Math.cos(el * rad) + (el >= 90 ? 0 : 3);
    cmp.clearRect(0, 0, 40, 40); cmp.strokeStyle = '#556'; cmp.lineWidth = 1; cmp.strokeRect(6, 6, 28, 28);
    cmp.strokeStyle = '#ffd88a'; cmp.fillStyle = '#ffd88a'; cmp.lineWidth = 2;
    const ex = 20 + len * Math.cos(az), ey = 20 + len * Math.sin(az); cmp.beginPath(); cmp.moveTo(20, 20); cmp.lineTo(ex, ey); cmp.stroke();
    cmp.beginPath(); cmp.arc(ex, ey, 2.5, 0, 7); cmp.fill();
  }
  for (const id of ['elev', 'az']) $(id).addEventListener('input', drawCompass);
  drawCompass();
  $('N').addEventListener('change', () => setN(+$('N').value));
  const syncMesh = () => { $('mesh').disabled = !$('decouple').checked; }; $('decouple').addEventListener('change', syncMesh); syncMesh();
  const setPlay = v => { R.playing = v; $('play').textContent = v ? 'Pause' : 'Play'; };
  $('play').onclick = () => setPlay(!R.playing);
  $('replay').onclick = async () => { await initSim(); setPlay(true); };
  $('next').onclick = () => { if (!R.busy) nextLetter(); };
  $('disturb').onclick = () => { if (!R.busy) disturb(); };
  document.querySelectorAll('#ui button').forEach(b => b.addEventListener('click', () => b.blur()));
  const toggleFull = () => {
    if (document.body.classList.contains('full')) { document.fullscreenElement && document.exitFullscreen(); document.body.classList.remove('full'); }
    else { document.body.classList.add('full'); document.documentElement.requestFullscreen?.().catch(() => {}); }
  };
  $('fs').onclick = toggleFull;
  document.addEventListener('fullscreenchange', () => { if (!document.fullscreenElement) document.body.classList.remove('full'); });
  if (qp.get('full') === '1') document.body.classList.add('full', 'kiosk');   // exhibition: start with the panel hidden (pair with Chrome --kiosk)
  document.querySelectorAll('button[data-p]').forEach(b => b.onclick = () => app.preset(+b.dataset.p));
  window.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' && e.target.type === 'text') return;
    const k = e.key.toLowerCase();
    if (k === 'f') toggleFull();
    else if (e.key === 'Escape') document.body.classList.remove('full');
    else if (e.key === ' ') { e.preventDefault(); $('play').click(); }
    else if (k === 'r') $('replay').click();
    else if (k === 'n') $('next').click();
    else if (k === 'd') $('disturb').click();
    else if (k === 'h') $('hudOn').checked = !$('hudOn').checked;
    else if (k === 'c') { $('clockOn').checked = !$('clockOn').checked; $('clockOn').dispatchEvent(new Event('change')); }
    else if (k === 'g') $('god').checked = !$('god').checked;
    else if (k === 'o') $('orbit').checked = !$('orbit').checked;
    else if (e.key >= '1' && e.key <= '3') app.preset(+e.key);
  });

  // clock <-> list switching and debug API
  $('clockOn').addEventListener('change', () => { if (clockOn()) { clk.key = clockKey(); clk.last = null; } else { R.listStr = null; } });
  rx.ambient = { state: () => ({ simT: AMB.simT, n: AMB.modes.length, modes: AMB.modes.map(m => [m.kx, m.ky, +m.T.toFixed(1)]) }), ratio: async () => { const s = await readStats(); return { ratio: Math.sqrt(s.e2) / R.rmsStar, nan: s.nan }; } };
  window.clock = {
    setTime: s => { const m = /^(\d{1,2}):?(\d{2})(?::?(\d{2}))?$/.exec(s); if (!m) throw new Error('HH:MM[:SS]'); const d = new Date(); d.setHours(+m[1], +m[2], +(m[3] || 0), 0); clk.off = d.getTime() - Date.now(); return shownStr(); },
    advance: min => { clk.off += min * 60000; return shownStr(); },
    state: () => ({ on: clockOn(), time: shownStr(), key: clk.key, ticks: clk.ticks, offsetMs: clk.off, layout: $('clockLayout').value, sec: showSec(), h12: $('clock12').value, N: S.N, resid: R.resid, nan: R.nan, fps: app.fps }),
  };

  // ---------- saved parameters (localStorage) ----------
  // Saved: every id'd range/checkbox/text/select in #ui + camera targets (yaw/pitch/dist) + look-at offset (via its sliders).
  // Not saved: clock time offset, panel open/fullscreen state, running state, HUD text.
  const PKEY = 'CausticTankClock.params.v1', PVER = 1;
  const pctl = [...document.querySelectorAll('#ui input[id], #ui select[id]')].filter(el => ['range', 'checkbox', 'text'].includes(el.type) || el.tagName === 'SELECT');
  const pdef = {}; for (const el of pctl) pdef[el.id] = el.type === 'checkbox' ? el.checked : (el.type === 'range' ? +el.value : el.value);
  const pdefCam = { ty: cam.ty, tp: cam.tp, td: cam.td }; app.paramDefaults = { params: pdef, camera: pdefCam };
  const psStatus = $('psStatus'), say = t => { psStatus.textContent = t; };
  const stamp = () => new Date().toTimeString().slice(0, 8);
  const lsGet = () => { try { return localStorage.getItem(PKEY); } catch (e) { return null; } };
  const lsSet = v => { try { localStorage.setItem(PKEY, v); return true; } catch (e) { return false; } };
  const lsDel = () => { try { localStorage.removeItem(PKEY); } catch (e) {} };
  const collect = () => { const params = {}; for (const el of pctl) params[el.id] = el.type === 'checkbox' ? el.checked : (el.type === 'range' ? +el.value : el.value);
    return { version: PVER, params, camera: { ty: +((((cam.ty % 360) + 360) % 360).toFixed(2)), tp: +cam.tp.toFixed(2), td: +cam.td.toFixed(3) } }; };
  // diff vs. the page defaults: only intentional changes are stored, so later default changes still reach untouched settings
  const diffOf = full => {
    const params = {}; for (const [id, v] of Object.entries(full.params)) { const d = pdef[id]; if (typeof v === 'number' ? Math.abs(v - d) > 1e-9 : v !== d) params[id] = v; }
    const camera = {}, c = full.camera, wrap = x => ((x % 360) + 540) % 360 - 180;
    if (!full.params.orbit && Math.abs(wrap(c.ty - pdefCam.ty)) > 1e-3) camera.ty = c.ty;   // with auto-orbit on, yaw is never stored
    if (Math.abs(c.tp - pdefCam.tp) > 1e-3) camera.tp = c.tp;
    if (Math.abs(c.td - pdefCam.td) > 1e-3) camera.td = c.td;
    const o = { version: PVER }; if (Object.keys(params).length) o.params = params; if (Object.keys(camera).length) o.camera = camera;
    return Object.keys(o).length > 1 ? o : null;
  };
  const storeDiff = () => { const d = diffOf(collect()); if (!d) { lsDel(); return 'nothing to save (all defaults)'; } return lsSet(JSON.stringify(d)) ? null : 'save failed (storage unavailable)'; };
  // validate + apply; returns { ok, bad } counts. fire: dispatch input/change so live handlers run (import); at startup the normal startup path reads the values.
  function applyParams(o, fire) {
    if (!o || typeof o !== 'object' || o.version !== PVER || (o.params !== undefined && (typeof o.params !== 'object' || !o.params))) throw new Error('bad format / version');
    let ok = 0, bad = 0;
    for (const [id, v] of Object.entries(o.params || {})) {
      const el = pctl.find(e => e.id === id); if (!el) { bad++; continue; }
      let good = false;
      if (el.type === 'checkbox') { if (typeof v === 'boolean') { el.checked = v; good = true; } }
      else if (el.type === 'range') { if (typeof v === 'number' && isFinite(v) && v >= +el.min - 1e-9 && v <= +el.max + 1e-9) { el.value = v; good = true; } }
      else if (el.tagName === 'SELECT') { const opt = [...el.options].find(x => x.value === String(v)); if (opt) { el.value = opt.value; good = true; } }
      else if (typeof v === 'string' && v.length <= 200) { el.value = v; good = true; }
      if (!good) { bad++; continue; } ok++;
      if (fire) { el.dispatchEvent(new Event('input', { bubbles: true })); el.dispatchEvent(new Event('change', { bubbles: true })); }
    }
    const c = o.camera;
    if (c && typeof c === 'object') {
      const nz = (x, lo, hi) => typeof x === 'number' && isFinite(x) && x >= lo && x <= hi;
      if (c.ty !== undefined && $('orbit').checked && !fire) { /* auto-orbit on: yaw is not restored */ } else if (nz(c.ty, -1e6, 1e6)) { cam.ty = c.ty; ok++; } else if (c.ty !== undefined) bad++;
      if (nz(c.tp, 4, 89.5)) { cam.tp = c.tp; ok++; } else if (c.tp !== undefined) bad++;
      if (nz(c.td, 1.2, 6)) { cam.td = c.td; ok++; } else if (c.td !== undefined) bad++;
    }
    readTargetSliders(); if (!fire) app.snapCam();
    return { ok, bad };
  }
  let noSave = false, saveTimer = 0, lastJson = '';
  const writeNow = (label = 'saved') => { if (noSave) return; lastJson = JSON.stringify(collect().params); const err = storeDiff(); say(err || `${label} ${stamp()}`); };
  const saveSoon = () => { clearTimeout(saveTimer); saveTimer = setTimeout(() => writeNow('auto-saved'), 300); };
  { const ui = $('ui'); ui.addEventListener('input', saveSoon); ui.addEventListener('change', saveSoon); }
  canvas.addEventListener('pointerup', saveSoon); canvas.addEventListener('wheel', saveSoon, { passive: true });
  document.querySelectorAll('button[data-p]').forEach(b => b.addEventListener('click', saveSoon));
  window.addEventListener('keydown', e => { if (!(e.target.tagName === 'INPUT' && e.target.type === 'text')) saveSoon(); });   // keyboard toggles (h/g/o/c/1-3) fire no events
  setInterval(() => { if (JSON.stringify(collect().params) !== lastJson) saveSoon(); }, 1000);
  const flush = () => { if (!noSave) { clearTimeout(saveTimer); try { storeDiff(); } catch (e) {} } };
  window.addEventListener('pagehide', flush); document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'hidden') flush(); });
  $('psSave').onclick = () => { clearTimeout(saveTimer); writeNow('saved'); };
  $('psReset').onclick = () => { noSave = true; clearTimeout(saveTimer); lsDel(); const u = new URL(location.href); u.searchParams.delete('reset'); location.replace(u.toString()); };
  $('psExport').onclick = () => { const blob = new Blob([JSON.stringify(collect(), null, 2)], { type: 'application/json' }), a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = 'CausticTankClock-params.json'; document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 1000); say('exported ' + stamp()); };
  const fileIn = document.createElement('input'); fileIn.type = 'file'; fileIn.accept = 'application/json,.json'; fileIn.style.display = 'none'; document.body.appendChild(fileIn);
  app.importParams = text => { const r = applyParams(JSON.parse(text), true); syncMesh(); drawCompass(); clearTimeout(saveTimer); writeNow('imported+saved'); say(psStatus.textContent + ` (${r.ok} ok, ${r.bad} ignored)`); return r; };
  fileIn.onchange = async () => { const f = fileIn.files[0]; fileIn.value = ''; if (!f) return; try { app.importParams(await f.text()); } catch (e) { say('import failed: ' + e.message); } };
  $('psImport').onclick = () => fileIn.click();
  document.querySelectorAll('#psBtns button').forEach(b => b.addEventListener('click', () => b.blur()));
  if (qp.get('reset') === '1') { lsDel(); say('stored params cleared (?reset=1)'); }
  else { const raw = lsGet();
    if (raw !== null) { try { const r = applyParams(JSON.parse(raw), false); say(`restored (${r.ok} ok, ${r.bad} ignored)`); }
      catch (e) { console.warn('param restore failed, using defaults', e); lsDel(); applyParams({ version: PVER, params: pdef, camera: pdefCam }, false); say('stored params invalid: reset to defaults'); } } }
  syncMesh(); drawCompass(); lastJson = JSON.stringify(collect().params);
  { const n0 = [128, 256, 512].includes(+qp.get('n')) ? +qp.get('n') : (+$('N').value || 256); $('N').value = n0; await setN(n0); }
  requestAnimationFrame(frame);
  app.ready = true;
}
main().catch(e => { document.getElementById('fps').textContent = 'ERROR: ' + e.message; console.error(e); });
