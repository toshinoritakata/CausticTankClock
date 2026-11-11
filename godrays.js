// God-ray light volume: V(x,y,z) = caustic of the current surface hT focused at depth z (z in [0,D], NZ slices, slice NZ-1 == floor caustic).
// Same exact vector-Snell forward splat as solver.js (`splat`), but the ray length is (z + hh)/(-tz) instead of (D + hh)/(-tz).
// Splat grid is M x M (M <= N, N/M integer); each volume texel launches `sub*sub` rays from the height field.  Accumulation in u32 atomics
// (fixed point, same SCALE as the solver), then separable Gaussian blur (X: accum -> tmp, Y: tmp -> r32float 3D texture).
export const GOD_SCALE = 65536;
export const godWGSL = (N) => {
  const bil = (name, buf) => `fn ${name}(px:f32,py:f32)->f32{
  let fx=floor(px); let fy=floor(py); let ax=px-fx; let ay=py-fy;
  let x0=i32(fx)&(NI-1); let y0=i32(fy)&(NI-1); let x1=(x0+1)&(NI-1); let y1=(y0+1)&(NI-1);
  return (${buf}[u32(y0*NI+x0)]*(1.0-ax)+${buf}[u32(y0*NI+x1)]*ax)*(1.0-ay)+(${buf}[u32(y1*NI+x0)]*(1.0-ax)+${buf}[u32(y1*NI+x1)]*ax)*ay;}`;
  return /* wgsl */`
const NI: i32 = ${N};
const SCALE: f32 = ${GOD_SCALE}.0;
struct G { D:f32, eta:f32, M:u32, NZ:u32, sub:u32, S:f32, sig:f32, rad:i32, amp:f32, tm:f32, dix:f32, diy:f32, diz:f32, fadeW:f32 }
@group(0) @binding(0) var<uniform> g: G;
@group(0) @binding(1) var<storage, read> hx: array<f32>;
@group(0) @binding(2) var<storage, read> hy: array<f32>;
@group(0) @binding(3) var<storage, read> hT: array<f32>;
@group(0) @binding(4) var<storage, read_write> accum: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> tmp: array<f32>;
@group(0) @binding(6) var volOut: texture_storage_3d<r32float, write>;
${bil('bHx', 'hx')}
${bil('bHy', 'hy')}
${bil('bH', 'hT')}
fn vidx(x:i32, y:i32, z:u32)->u32 { let M = i32(g.M); return z * g.M * g.M + u32(y & (M - 1)) * g.M + u32(x & (M - 1)); }
@compute @workgroup_size(8,8,1) fn vclear(@builtin(global_invocation_id) gi:vec3u) {
  if (gi.x >= g.M || gi.y >= g.M || gi.z >= g.NZ) { return; }
  atomicStore(&accum[vidx(i32(gi.x), i32(gi.y), gi.z)], 0u);
}
// rays that have left the tank footprint (through a side wall) before reaching depth z fade out; fadeW = 0 -> periodic wrap (same as the floor caustic)
fn fadeMask(fx:f32, fy:f32)->f32 {
  if (g.fadeW <= 0.0) { return 1.0; }
  let px = fx / f32(g.M); let py = fy / f32(g.M); let f = g.fadeW;
  return smoothstep(0.0, f, px) * smoothstep(0.0, f, 1.0 - px) * smoothstep(0.0, f, py) * smoothstep(0.0, f, 1.0 - py);
}
fn splat1(fx:f32, fy:f32, z:u32, w:f32) {
  if (!(abs(fx) < 1.0e6 && abs(fy) < 1.0e6)) { return; }
  let fx0 = floor(fx); let fy0 = floor(fy); let ax = fx - fx0; let ay = fy - fy0;
  let X0 = i32(fx0); let Y0 = i32(fy0); let s = w * SCALE;
  atomicAdd(&accum[vidx(X0, Y0, z)], u32(s * (1.0 - ax) * (1.0 - ay) + 0.5));
  atomicAdd(&accum[vidx(X0 + 1, Y0, z)], u32(s * ax * (1.0 - ay) + 0.5));
  atomicAdd(&accum[vidx(X0, Y0 + 1, z)], u32(s * (1.0 - ax) * ay + 0.5));
  atomicAdd(&accum[vidx(X0 + 1, Y0 + 1, z)], u32(s * ax * ay + 0.5));
}
@compute @workgroup_size(8,8,1) fn vsplat(@builtin(global_invocation_id) gi:vec3u) {
  if (gi.x >= g.M || gi.y >= g.M || gi.z >= g.NZ) { return; }
  let z = g.D * f32(gi.z) / f32(max(g.NZ, 2u) - 1u);
  let sub = i32(g.sub); let w = 1.0 / f32(sub * sub); let cen = f32(sub - 1) / (2.0 * f32(sub));
  let Mf = f32(g.M);
  for (var sy = 0; sy < sub; sy++) { for (var sx = 0; sx < sub; sx++) {
    let ux = f32(gi.x) + f32(sx) / f32(sub) - cen; let uy = f32(gi.y) + f32(sy) / f32(sub) - cen;
    let px = ux * g.S; let py = uy * g.S;
    let gx = bHx(px, py); let gy = bHy(px, py); let hh = bH(px, py);
    let inv = 1.0 / sqrt(1.0 + gx*gx + gy*gy);
    let n = vec3f(-gx * inv, -gy * inv, inv);
    let di = vec3f(g.dix, g.diy, g.diz);                       // oblique incident direction (solver frame: x,y floor, z up)
    let c = -dot(di, n); let kk = 1.0 - g.eta*g.eta*(1.0 - c*c);
    let f = g.eta * c - sqrt(kk);
    let t = g.eta * di + f * n;                                // exact vector Snell
    let len = (z + hh) / (-t.z);
    let fx = ux + len * t.x * Mf; let fy = uy + len * t.y * Mf;
    splat1(fx, fy, gi.z, w * fadeMask(fx, fy));
  } }
}
fn gw(k:i32)->f32 { return exp(-0.5 * f32(k*k) / (g.sig * g.sig)); }
@compute @workgroup_size(8,8,1) fn vblurX(@builtin(global_invocation_id) gi:vec3u) {
  if (gi.x >= g.M || gi.y >= g.M || gi.z >= g.NZ) { return; }
  var s = 0.0; var ws = 0.0;
  for (var k = -g.rad; k <= g.rad; k++) { let w = gw(k); ws += w; s += w * f32(atomicLoad(&accum[vidx(i32(gi.x) + k, i32(gi.y), gi.z)])); }
  tmp[vidx(i32(gi.x), i32(gi.y), gi.z)] = s / (ws * SCALE);
}
fn h13(p:vec3f)->f32 { var q=fract(p*0.1031); q+=dot(q,q.zyx+31.32); return fract((q.x+q.y)*q.z); }
fn vnoise(p:vec3f)->f32 {
  let i=floor(p); let f=fract(p); let w=f*f*(3.0-2.0*f);
  return mix(mix(mix(h13(i),h13(i+vec3f(1,0,0)),w.x), mix(h13(i+vec3f(0,1,0)),h13(i+vec3f(1,1,0)),w.x), w.y),
             mix(mix(h13(i+vec3f(0,0,1)),h13(i+vec3f(1,0,1)),w.x), mix(h13(i+vec3f(0,1,1)),h13(i+vec3f(1,1,1)),w.x), w.y), w.z);
}
@compute @workgroup_size(8,8,1) fn vblurY(@builtin(global_invocation_id) gi:vec3u) {
  if (gi.x >= g.M || gi.y >= g.M || gi.z >= g.NZ) { return; }
  var s = 0.0; var ws = 0.0;
  for (var k = -g.rad; k <= g.rad; k++) { let w = gw(k); ws += w; s += w * tmp[vidx(i32(gi.x), i32(gi.y) + k, gi.z)]; }
  var dust = 1.0;
  if (g.amp > 0.0) { // faint animated particulate modulation (world-space, baked into the volume)
    let zz = g.D * f32(gi.z) / f32(max(g.NZ, 2u) - 1u);
    let p = vec3f((f32(gi.x) + 0.5) / f32(g.M) - 0.5, -zz, (f32(gi.y) + 0.5) / f32(g.M) - 0.5);
    let nq = vnoise(p * vec3f(14.0, 9.0, 14.0) + vec3f(0.0, g.tm * 0.05, g.tm * 0.03)) * 0.65 + vnoise(p * vec3f(33.0, 20.0, 33.0) + vec3f(g.tm * 0.04, 0.0, 0.0)) * 0.35;
    dust = 1.0 + g.amp * (nq - 0.5) * 2.0;
  }
  textureStore(volOut, vec3u(gi.x, gi.y, gi.z), vec4f(s / ws * dust, 0.0, 0.0, 0.0));
}
`;
};
