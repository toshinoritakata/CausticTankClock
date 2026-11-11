// Compute-pass encoders shared by the app loop and the debug hooks. Pure functions of (pass, ctx): no module state.
// `c` is a per-N context made by gpu.js makeCtx(); `p` the solver params from app.js paramsFinal().

// one workgroup per 8x8 tile / a single workgroup
export const d8 = (pass, c, pipe) => { pass.setPipeline(pipe); pass.dispatchWorkgroups(c.N / 8, c.N / 8); };
export const d1 = (pass, pipe) => { pass.setPipeline(pipe); pass.dispatchWorkgroups(1); };

// 2-D FFT: rows forward, column multiply (mode pipeline), rows inverse
export function fft3(pass, c, colsPipe) {
  pass.setPipeline(c.P.rowsFwd); pass.dispatchWorkgroups(c.N);
  pass.setPipeline(colsPipe); pass.dispatchWorkgroups(c.N);
  pass.setPipeline(c.P.rowsInv); pass.dispatchWorkgroups(c.N);
}

// Monge-Ampere solve for the designed surface h*
export function encodeSolve(pass, c, p) {
  pass.setBindGroup(0, c.bg.pre); d8(pass, c, c.P.mixTex); fft3(pass, c, c.P.colsPre);
  d1(pass, c.P.red0); d8(pass, c, c.P.prepT); d1(pass, c.P.red1); d8(pass, c, c.P.normT);
  pass.setBindGroup(0, c.bg.first); d8(pass, c, c.P.initT); fft3(pass, c, c.P.colsPoisson);
  pass.setBindGroup(0, c.bg.iter);
  for (let i = 0; i < p.iters; i++) { d8(pass, c, c.P.maWarp); fft3(pass, c, c.P.colsPoisson); }
}

// n damped-wave steps (ping-pong E / V buffers via c.cur)
export function encodeSteps(pass, c, n) {
  for (let i = 0; i < n; i++) { pass.setPipeline(c.rp.step); pass.setBindGroup(0, c.bgP[c.cur]); pass.dispatchWorkgroups(c.N / 8, c.N / 8); c.cur ^= 1; }
}

// caustic of the surface currently in B.hT: grad, splat, blur -> B.C
export function encodeCaustic(pass, c, P = c.P) {
  pass.setBindGroup(0, c.bg.R); d8(pass, c, P.grad); d8(pass, c, P.clearAcc); d8(pass, c, P.splat);
  pass.setBindGroup(0, c.bg.c); d8(pass, c, P.accumToC); fft3(pass, c, P.colsC);
}

// god-ray light volume
export function encodeVolume(pass, c) {
  const g = c.god, wx = g.M / 8, z = g.NZ; pass.setBindGroup(0, g.bg);
  for (const e of ['vclear', 'vsplat', 'vblurX', 'vblurY']) { pass.setPipeline(g.gp[e]); pass.dispatchWorkgroups(wx, wx, z); }
}

// run a small one-off compute pass (retarget / kick)
export function submitKick(device, c) {
  const enc = device.createCommandEncoder(), ps = enc.beginComputePass();
  ps.setPipeline(c.rp.addKick); ps.setBindGroup(0, c.bgP[1 - c.cur]); ps.dispatchWorkgroups(c.N / 8, c.N / 8); ps.end();
  device.queue.submit([enc.finish()]);
}
