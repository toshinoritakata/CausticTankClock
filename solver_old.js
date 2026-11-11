// Copied verbatim from ../caustics_gpu/app.js (compute solver WGSL: FFT Poisson / Monge-Ampere / forward splat).
export const SCALE = 65536;
export function computeWGSL(N) {
  const bilFn = (name, buf) => `fn ${name}(x:i32,y:i32,ax:f32,ay:f32)->f32{
  let x1=(x+1)&(NI-1); let y1=(y+1)&(NI-1);
  return (${buf}[u32(y*NI+x)]*(1.0-ax)+${buf}[u32(y*NI+x1)]*ax)*(1.0-ay)+(${buf}[u32(y1*NI+x)]*(1.0-ax)+${buf}[u32(y1*NI+x1)]*ax)*ay;}`;
  return /* wgsl */`
const N: u32 = ${N}u;
const NI: i32 = ${N};
const HALF: u32 = ${N / 2}u;
const WGS: u32 = ${Math.min(256, N / 2)}u;
const PI: f32 = 3.14159265358979;
const SCALE: f32 = ${SCALE}.0;
struct Params { D:f32, k:f32, contrast:f32, morph:f32, sigPre:f32, sigC:f32, sub:f32, nw:f32, beta:f32, p0:f32, p1:f32, p2:f32 }
struct Mix { a:f32, b:f32, p0:f32, p1:f32 }
@group(0) @binding(0) var<uniform> prm: Params;
@group(0) @binding(1) var<uniform> mx: Mix;
@group(0) @binding(2) var<storage, read_write> cbuf: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> field: array<f32>;
@group(0) @binding(4) var<storage, read_write> Tb: array<f32>;
@group(0) @binding(5) var<storage, read_write> hxb: array<f32>;
@group(0) @binding(6) var<storage, read_write> hyb: array<f32>;
@group(0) @binding(7) var<storage, read_write> accum: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> stats: array<f32>;
@group(0) @binding(9) var texA: texture_2d<f32>;
@group(0) @binding(10) var texB: texture_2d<f32>;
@group(0) @binding(11) var<storage, read_write> tw: array<vec2f>; // [0..HALF]: exp(-2 pi i m/N); [HALF+1+x]: 4 sin^2(pi x/N)

override MODE: u32 = 0u;
override INV: bool = false;

fn cmul(a:vec2f,b:vec2f)->vec2f { return vec2f(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x); }

var<workgroup> sh: array<vec2f, ${2 * N}>;
// Stockham radix-2 autosort FFT of one line held in shared memory at offset src0. Returns offset of the result.
fn fft_sh(lid:u32, inv:bool, src0:u32)->u32 {
  var src = src0; var ns = 1u;
  loop {
    if (ns >= N) { break; }
    let dsto = N - src;
    for (var j = lid; j < HALF; j += WGS) {
      let k = j & (ns - 1u);
      var w = tw[k * (HALF / ns)];
      if (inv) { w.y = -w.y; }
      let v0 = sh[src + j];
      let v1 = cmul(sh[src + j + HALF], w);
      let d = ((j - k) << 1u) + k;
      sh[dsto + d] = v0 + v1;
      sh[dsto + d + ns] = v0 - v1;
    }
    workgroupBarrier();
    src = dsto; ns = ns << 1u;
  }
  return src;
}
fn frq(i:u32)->f32 { return select(f32(i) - f32(N), f32(i), i <= HALF); }
fn lamS(i:u32)->f32 { return tw[HALF + 1u + i].x; }
fn mult(x:u32, y:u32)->f32 {
  if (MODE == 0u) { // Poisson: 1/(-lam*k), lam = -(4sin^2+4sin^2)*N^2 (5-point Laplacian eigenvalue), DC zeroed
    if (x == 0u && y == 0u) { return 0.0; }
    return 1.0 / ((lamS(x) + lamS(y)) * f32(N * N) * prm.k);
  }
  let s = select(prm.sigC, prm.sigPre, MODE == 1u);
  let kx = 2.0 * PI * frq(x) / f32(N); let ky = 2.0 * PI * frq(y) / f32(N);
  return exp(-0.5 * s * s * (kx*kx + ky*ky));
}

@compute @workgroup_size(WGS) fn rowsFwd(@builtin(local_invocation_index) lid:u32, @builtin(workgroup_id) wg:vec3u) {
  let r = wg.x;
  for (var i = lid; i < N; i += WGS) { sh[i] = cbuf[r * N + i]; }
  workgroupBarrier();
  let s = fft_sh(lid, false, 0u);
  for (var i = lid; i < N; i += WGS) { cbuf[r * N + i] = sh[s + i]; }
}
// columns: forward FFT, multiply by spectral multiplier, inverse FFT (all in shared memory)
@compute @workgroup_size(WGS) fn colsMul(@builtin(local_invocation_index) lid:u32, @builtin(workgroup_id) wg:vec3u) {
  let c = wg.x;
  for (var i = lid; i < N; i += WGS) { sh[i] = cbuf[i * N + c]; }
  workgroupBarrier();
  let s = fft_sh(lid, false, 0u);
  for (var i = lid; i < N; i += WGS) { sh[s + i] = sh[s + i] * mult(c, i); }
  workgroupBarrier();
  let s2 = fft_sh(lid, true, s);
  for (var i = lid; i < N; i += WGS) { cbuf[i * N + c] = sh[s2 + i]; }
}
// rows: inverse FFT, write real part blended into field: field = a*re + b*field
@compute @workgroup_size(WGS) fn rowsInv(@builtin(local_invocation_index) lid:u32, @builtin(workgroup_id) wg:vec3u) {
  let r = wg.x;
  for (var i = lid; i < N; i += WGS) { sh[i] = cbuf[r * N + i]; }
  workgroupBarrier();
  let s = fft_sh(lid, true, 0u);
  for (var i = lid; i < N; i += WGS) { let idx = r * N + i; var v = mx.a * sh[s + i].x; if (mx.b != 0.0) { v += mx.b * field[idx]; } field[idx] = v; }
}
// plain 1D FFT passes for unit tests (INV override)
@compute @workgroup_size(WGS) fn testRows(@builtin(local_invocation_index) lid:u32, @builtin(workgroup_id) wg:vec3u) {
  let r = wg.x;
  for (var i = lid; i < N; i += WGS) { sh[i] = cbuf[r * N + i]; }
  workgroupBarrier();
  let s = fft_sh(lid, INV, 0u);
  for (var i = lid; i < N; i += WGS) { cbuf[r * N + i] = sh[s + i]; }
}
@compute @workgroup_size(WGS) fn testCols(@builtin(local_invocation_index) lid:u32, @builtin(workgroup_id) wg:vec3u) {
  let c = wg.x;
  for (var i = lid; i < N; i += WGS) { sh[i] = cbuf[i * N + c]; }
  workgroupBarrier();
  let s = fft_sh(lid, INV, 0u);
  for (var i = lid; i < N; i += WGS) { cbuf[i * N + c] = sh[s + i]; }
}

fn gray(c:vec4f)->f32 { return 0.299*c.r + 0.587*c.g + 0.114*c.b; }
@compute @workgroup_size(8,8) fn mixTex(@builtin(global_invocation_id) g:vec3u) {
  let p = vec2i(g.xy);
  let a = gray(textureLoad(texA, p, 0)); let b = gray(textureLoad(texB, p, 0));
  cbuf[g.y * N + g.x] = vec2f(mix(a, b, prm.morph), 0.0);
}
var<workgroup> red: array<f32, 256>;
// MODE 0: mean(field) -> stats[0]; 1: mean(Tb) -> stats[1]; 2: max(|hx|,|hy|) -> stats[2]
@compute @workgroup_size(256) fn reduce(@builtin(local_invocation_index) lid:u32) {
  var acc = 0.0;
  for (var i = lid; i < N * N; i += 256u) {
    if (MODE == 0u) { acc += field[i]; }
    else if (MODE == 1u) { acc += Tb[i]; }
    else { acc = max(acc, max(abs(hxb[i]), abs(hyb[i]))); }
  }
  red[lid] = acc;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid < s) { if (MODE == 2u) { red[lid] = max(red[lid], red[lid + s]); } else { red[lid] += red[lid + s]; } }
    workgroupBarrier();
  }
  if (lid == 0u) { stats[MODE] = select(red[0] / f32(N * N), red[0], MODE == 2u); }
}
@compute @workgroup_size(8,8) fn prepT(@builtin(global_invocation_id) g:vec3u) {
  let i = g.y * N + g.x; let m = stats[0] + 1e-9;
  Tb[i] = max(0.05, 1.0 + prm.contrast * (field[i] / m - 1.0));
}
@compute @workgroup_size(8,8) fn normT(@builtin(global_invocation_id) g:vec3u) {
  let i = g.y * N + g.x; Tb[i] = Tb[i] / stats[1];
}
@compute @workgroup_size(8,8) fn initT(@builtin(global_invocation_id) g:vec3u) {
  let i = g.y * N + g.x; cbuf[i] = vec2f(Tb[i] - 1.0, 0.0);
}
${bilFn('bilT', 'Tb')}
${bilFn('bilHx', 'hxb')}
${bilFn('bilHy', 'hyb')}
${bilFn('bilH', 'field')}
// Monge-Ampere backward step: land position x + k*grad h, sample T there, write RHS (1 - 1/T_warped)
@compute @workgroup_size(8,8) fn maWarp(@builtin(global_invocation_id) g:vec3u) {
  let x = i32(g.x); let y = i32(g.y); let i = g.y * N + g.x;
  let xp = (x + 1) & (NI - 1); let xm = (x + NI - 1) & (NI - 1); let yp = (y + 1) & (NI - 1); let ym = (y + NI - 1) & (NI - 1);
  let s = f32(N) * 0.5;
  let hx = (field[u32(y * NI + xp)] - field[u32(y * NI + xm)]) * s;
  let hy = (field[u32(yp * NI + x)] - field[u32(ym * NI + x)]) * s;
  let fx = f32(x) + f32(N) * prm.k * hx; let fy = f32(y) + f32(N) * prm.k * hy;
  let fx0 = floor(fx); let fy0 = floor(fy);
  let tw_ = bilT(i32(fx0) & (NI - 1), i32(fy0) & (NI - 1), fx - fx0, fy - fy0);
  cbuf[i] = vec2f(1.0 - 1.0 / tw_, 0.0);
}
@compute @workgroup_size(8,8) fn grad(@builtin(global_invocation_id) g:vec3u) {
  let x = i32(g.x); let y = i32(g.y); let i = g.y * N + g.x;
  let xp = (x + 1) & (NI - 1); let xm = (x + NI - 1) & (NI - 1); let yp = (y + 1) & (NI - 1); let ym = (y + NI - 1) & (NI - 1);
  let s = f32(N) * 0.5;
  hxb[i] = (field[u32(y * NI + xp)] - field[u32(y * NI + xm)]) * s;
  hyb[i] = (field[u32(yp * NI + x)] - field[u32(ym * NI + x)]) * s;
}
@compute @workgroup_size(8,8) fn clearAcc(@builtin(global_invocation_id) g:vec3u) { atomicStore(&accum[g.y * N + g.x], 0u); }
@compute @workgroup_size(8,8) fn accumToC(@builtin(global_invocation_id) g:vec3u) {
  cbuf[g.y * N + g.x] = vec2f(f32(atomicLoad(&accum[g.y * N + g.x])) / SCALE, 0.0);
}
@compute @workgroup_size(8,8) fn accumCopy(@builtin(global_invocation_id) g:vec3u) {
  field[g.y * N + g.x] = f32(atomicLoad(&accum[g.y * N + g.x])) / SCALE;
}
fn splat1(fx:f32, fy:f32, w:f32) {
  if (!(abs(fx) < 1.0e6 && abs(fy) < 1.0e6)) { return; }
  let fx0 = floor(fx); let fy0 = floor(fy); let ax = fx - fx0; let ay = fy - fy0;
  let X0 = i32(fx0) & (NI - 1); let X1 = (X0 + 1) & (NI - 1); let Y0 = i32(fy0) & (NI - 1); let Y1 = (Y0 + 1) & (NI - 1);
  let s = w * SCALE;
  atomicAdd(&accum[u32(Y0 * NI + X0)], u32(s * (1.0 - ax) * (1.0 - ay) + 0.5));
  atomicAdd(&accum[u32(Y0 * NI + X1)], u32(s * ax * (1.0 - ay) + 0.5));
  atomicAdd(&accum[u32(Y1 * NI + X0)], u32(s * (1.0 - ax) * ay + 0.5));
  atomicAdd(&accum[u32(Y1 * NI + X1)], u32(s * ax * ay + 0.5));
}
// forward caustic: refract vertical rays through the surface (n=nw) to floor depth D, bilinear atomic splat
@compute @workgroup_size(8,8) fn splat(@builtin(global_invocation_id) g:vec3u) {
  let x = i32(g.x); let y = i32(g.y);
  let sub = i32(prm.sub); let w = 1.0 / f32(sub * sub); let eta = 1.0 / prm.nw;
  let cen = f32(sub - 1) / (2.0 * f32(sub));
  for (var sy = 0; sy < sub; sy++) { for (var sx = 0; sx < sub; sx++) {
    let ox = f32(sx) / f32(sub); let oy = f32(sy) / f32(sub);
    let gx = bilHx(x, y, ox, oy); let gy = bilHy(x, y, ox, oy); let hh = bilH(x, y, ox, oy);
    let inv = 1.0 / sqrt(1.0 + gx*gx + gy*gy);
    let nx = -gx * inv; let ny = -gy * inv; let nz = inv;
    let cosI = nz; let kk = 1.0 - eta*eta*(1.0 - cosI*cosI);
    let f = eta * cosI - sqrt(kk);
    let tx = f * nx; let ty = f * ny; let tz = -eta + f * nz;
    let len = (prm.D + hh) / (-tz);
    splat1(f32(x) + ox - cen + len * tx * f32(N), f32(y) + oy - cen + len * ty * f32(N), w);
  } }
}
`;
}
