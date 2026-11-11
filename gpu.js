// WebGPU setup: device, shader compile helper, render pipelines + canvas targets, and the per-N compute context
// (solver + wave-relax + god-ray buffers/pipelines). Shader sources are passed in by app.js, which owns the cache-busted imports.

export const ST = GPUBufferUsage.STORAGE, CS = GPUBufferUsage.COPY_SRC, CD = GPUBufferUsage.COPY_DST, UN = GPUBufferUsage.UNIFORM;
const SAMPLES = 4;                  // MSAA
const WGSL_COMPUTE = GPUShaderStage.COMPUTE;

export async function initDevice(canvas, onError) {
  if (!navigator.gpu) throw new Error('WebGPU not available');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('no adapter');
  const device = await adapter.requestDevice({ requiredFeatures: adapter.features.has('float32-filterable') ? ['float32-filterable'] : [] });
  device.addEventListener('uncapturederror', e => { console.error('WebGPU uncaptured error: ' + e.error.message); onError(e.error.message); });
  device.lost.then(i => console.error('device lost: ' + i.message));
  const gctx = canvas.getContext('webgpu'), fmt = navigator.gpu.getPreferredCanvasFormat();
  gctx.configure({ device, format: fmt, alphaMode: 'opaque' });
  return { device, gctx, fmt };
}

export async function compileWGSL(device, code, label) {
  const m = device.createShaderModule({ code });
  const info = await m.getCompilationInfo();
  for (const x of info.messages) console.log(`WGSL ${label} ${x.type} line ${x.lineNum}: ${x.message}`);
  if (info.messages.some(x => x.type === 'error')) throw new Error('WGSL compile error in ' + label);
  return m;
}

// read a GPU buffer back as Float32Array (one-off staging buffer)
export async function readBuffer(device, buf, size) {
  const st = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | CD }), e = device.createCommandEncoder();
  e.copyBufferToBuffer(buf, 0, st, 0, size); device.queue.submit([e.finish()]);
  await st.mapAsync(GPUMapMode.READ); const o = new Float32Array(st.getMappedRange().slice(0)); st.unmap(); st.destroy(); return o;
}

// ---------- render pipelines + MSAA/depth targets ----------
export async function createRenderer({ device, gctx, fmt, canvas, qp, SCENE_WGSL, BG_WGSL }) {
  const sceneMod = await compileWGSL(device, SCENE_WGSL, 'scene'), bgMod = await compileWGSL(device, BG_WGSL, 'bg');
  const VF = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
  const bgl = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: VF, buffer: { type: 'uniform' } },
    ...[1, 2, 3, 4].map(b => ({ binding: b, visibility: VF, buffer: { type: 'read-only-storage' } })),
    { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '3d' } },
    { binding: 6, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
  ] });
  const vsamp = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge' });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
  const blend = { color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } };
  const mkRP = (mod, vs, fs, o = {}) => device.createRenderPipeline({
    layout, vertex: { module: mod, entryPoint: vs }, fragment: { module: mod, entryPoint: fs, targets: [{ format: fmt, blend: o.blend }] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: o.dw !== false, depthCompare: o.cmp || 'less' },
    multisample: { count: SAMPLES },
  });
  const RP = {
    bg: mkRP(bgMod, 'vs', 'fs', { dw: false, cmp: 'always' }),
    table: mkRP(sceneMod, 'vsTable', 'fsTable'), floor: mkRP(sceneMod, 'vsFloor', 'fsFloor'),
    water: mkRP(sceneMod, 'vsWater', 'fsWater', { blend }), glass: mkRP(sceneMod, 'vsGlass', 'fsGlass', { blend, dw: false }),
  };
  const ubuf = device.createBuffer({ size: 256 + 96, usage: UN | CD });   // view/light/look uniforms (written by app.js writeUniforms)
  const size = { w: 0, h: 0 };
  let msaaTex = null, depthTex = null;
  // resize the canvas + render targets to the element size (?w= / ?h= force a fixed size)
  function fitCanvas() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const fw = +qp.get('w'), fh = +qp.get('h');
    const w = fw || Math.max(64, Math.round(canvas.clientWidth * dpr)), h = fh || Math.max(64, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h || !msaaTex) {
      canvas.width = w; canvas.height = h; size.w = w; size.h = h;
      msaaTex?.destroy(); depthTex?.destroy();
      msaaTex = device.createTexture({ size: [w, h], sampleCount: SAMPLES, format: fmt, usage: GPUTextureUsage.RENDER_ATTACHMENT });
      depthTex = device.createTexture({ size: [w, h], sampleCount: SAMPLES, format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });
    }
  }
  // meshVerts = water mesh resolution (vertices per side)
  function encodeRender(enc, c, meshVerts) {
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: msaaTex.createView(), resolveTarget: gctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'discard', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
      depthStencilAttachment: { view: depthTex.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' },
    });
    rp.setBindGroup(0, c.renderBG);
    rp.setPipeline(RP.bg); rp.draw(3); rp.setPipeline(RP.table); rp.draw(6); rp.setPipeline(RP.floor); rp.draw(6);
    rp.setPipeline(RP.water); rp.draw((meshVerts - 1) * (meshVerts - 1) * 6); rp.setPipeline(RP.glass); rp.draw(6, 4); rp.end();
  }
  return { bgl, vsamp, ubuf, size, fitCanvas, encodeRender };
}

// ---------- per-N compute context: solver + relax + god-ray buffers/pipelines ----------
// shaders = { computeWGSL, RELAX_WGSL, godWGSL }; renderer = result of createRenderer
export function createContexts({ device, shaders, renderer }) {
  const cache = {};
  async function makeCtx(N) {
    if (cache[N]) return cache[N];
    const mod = await compileWGSL(device, shaders.computeWGSL(N), 'solver' + N);
    const entries = [
      { binding: 0, visibility: WGSL_COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: WGSL_COMPUTE, buffer: { type: 'uniform' } },
      ...[2, 3, 4, 5, 6, 7, 8].map(b => ({ binding: b, visibility: WGSL_COMPUTE, buffer: { type: 'storage' } })),
      { binding: 9, visibility: WGSL_COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 10, visibility: WGSL_COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 11, visibility: WGSL_COMPUTE, buffer: { type: 'storage' } },
    ];
    const bgl = device.createBindGroupLayout({ entries });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
    const P = solverPipelines(device, mod, layout);
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
    const rmod = await compileWGSL(device, shaders.RELAX_WGSL(N), 'relax' + N);
    const rEntries = [[0, 'uniform'], [1, 'read-only-storage'], [2, 'read-only-storage'], [3, 'storage'], [4, 'storage'], [5, 'read-only-storage'], [6, 'read-only-storage'], [7, 'storage'], [8, 'storage'], [9, 'uniform']]
      .map(([binding, type]) => ({ binding, visibility: WGSL_COMPUTE, buffer: { type } }));
    const rbgl = device.createBindGroupLayout({ entries: rEntries });
    const rlay = device.createPipelineLayout({ bindGroupLayouts: [rbgl] });
    const rp = {}; for (const e of ['step', 'compose', 'retarget', 'addKick', 'reduce']) rp[e] = device.createComputePipeline({ layout: rlay, compute: { module: rmod, entryPoint: e } });
    const E = [mk(nn * 4, ST | CS | CD), mk(nn * 4, ST | CS | CD)], Vb = [mk(nn * 4, ST | CS | CD), mk(nn * 4, ST | CS | CD)];
    const ambU = mk(192, UN | CD), hp = mk(nn * 4, ST | CS | CD), rstats = mk(16, ST | CS), rU = mk(32, UN | CD);
    const statStage = mk(16, GPUBufferUsage.MAP_READ | CD), stageB = mk(nn * 4, GPUBufferUsage.MAP_READ | CD), stageE = mk(nn * 4, GPUBufferUsage.MAP_READ | CD);
    const bgP = [0, 1].map(k => device.createBindGroup({ layout: rbgl, entries: [
      { binding: 0, resource: { buffer: rU } }, { binding: 1, resource: { buffer: E[k] } }, { binding: 2, resource: { buffer: Vb[k] } },
      { binding: 3, resource: { buffer: E[1 - k] } }, { binding: 4, resource: { buffer: Vb[1 - k] } }, { binding: 5, resource: { buffer: B.field } },
      { binding: 6, resource: { buffer: hp } }, { binding: 7, resource: { buffer: B.hT } }, { binding: 8, resource: { buffer: rstats } }, { binding: 9, resource: { buffer: ambU } }] }));
    const ctx = { N, P, B, bg, layout, texA, texB, renderBG: null, cv: [document.createElement('canvas'), document.createElement('canvas')], raw: [document.createElement('canvas'), document.createElement('canvas')], rp, E, Vb, hp, ambU, kV: hp, rstats, rU, statStage, stageB, stageE, bgP, cur: 0 };   // kV: the kick velocity is staged in hp

    // ----- god-ray light volume -----
    const gmod = await compileWGSL(device, shaders.godWGSL(N), 'god' + N);
    const gbgl = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: WGSL_COMPUTE, buffer: { type: 'uniform' } },
      ...[1, 2, 3].map(b => ({ binding: b, visibility: WGSL_COMPUTE, buffer: { type: 'read-only-storage' } })),
      { binding: 4, visibility: WGSL_COMPUTE, buffer: { type: 'storage' } }, { binding: 5, visibility: WGSL_COMPUTE, buffer: { type: 'storage' } },
      { binding: 6, visibility: WGSL_COMPUTE, storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '3d' } },
    ] });
    const glay = device.createPipelineLayout({ bindGroupLayouts: [gbgl] });
    ctx.god = { gp: {}, gU: mk(64, UN | CD), M: 0, NZ: 0, vol: null };
    for (const e of ['vclear', 'vsplat', 'vblurX', 'vblurY']) ctx.god.gp[e] = device.createComputePipeline({ layout: glay, compute: { module: gmod, entryPoint: e } });
    ctx.god.gbgl = gbgl;
    return cache[N] = ctx;
  }

  // (re)allocate the volume for the current volume-res setting / slice count NZ and rebuild the render bind group
  function ensureVol(c, volRes, nz) {
    const g = c.god, M = Math.min(c.N, volRes), NZ = Math.max(2, Math.round(nz));
    if (g.M === M && g.NZ === NZ) return;
    g.vol?.destroy(); g.accum?.destroy(); g.tmp?.destroy();
    g.M = M; g.NZ = NZ; g.sub = g.sub || 1;
    g.vol = device.createTexture({ size: [M, M, NZ], dimension: '3d', format: 'r32float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
    g.accum = device.createBuffer({ size: M * M * NZ * 4, usage: ST | CS | CD }); g.tmp = device.createBuffer({ size: M * M * NZ * 4, usage: ST | CS | CD });
    g.bg = device.createBindGroup({ layout: g.gbgl, entries: [
      { binding: 0, resource: { buffer: g.gU } }, { binding: 1, resource: { buffer: c.B.hx } }, { binding: 2, resource: { buffer: c.B.hy } }, { binding: 3, resource: { buffer: c.B.hT } },
      { binding: 4, resource: { buffer: g.accum } }, { binding: 5, resource: { buffer: g.tmp } }, { binding: 6, resource: g.vol.createView({ dimension: '3d' }) }] });
    c.renderBG = device.createBindGroup({ layout: renderer.bgl, entries: [
      { binding: 0, resource: { buffer: renderer.ubuf } }, { binding: 1, resource: { buffer: c.B.hT } }, { binding: 2, resource: { buffer: c.B.hx } },
      { binding: 3, resource: { buffer: c.B.hy } }, { binding: 4, resource: { buffer: c.B.C } },
      { binding: 5, resource: g.vol.createView({ dimension: '3d' }) }, { binding: 6, resource: renderer.vsamp }] });
  }
  return { makeCtx, ensureVol };
}

// the 12 solver compute entry points + the 3 column-multiply and 2 reduce specialisations
export function solverPipelines(device, mod, layout) {
  const pipe = (entryPoint, constants) => device.createComputePipeline({ layout, compute: { module: mod, entryPoint, constants } });
  const P = {};
  for (const e of ['rowsFwd', 'rowsInv', 'mixTex', 'prepT', 'normT', 'initT', 'maWarp', 'grad', 'clearAcc', 'accumToC', 'accumCopy', 'splat']) P[e] = pipe(e);
  P.colsPoisson = pipe('colsMul', { MODE: 0 }); P.colsPre = pipe('colsMul', { MODE: 1 }); P.colsC = pipe('colsMul', { MODE: 2 });
  P.red0 = pipe('reduce', { MODE: 0 }); P.red1 = pipe('reduce', { MODE: 1 });
  return P;
}
