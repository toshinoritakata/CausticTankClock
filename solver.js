// Copied verbatim from ../caustics_gpu/app.js (compute solver WGSL: FFT Poisson / Monge-Ampere / forward splat); local change: the forward splat uses Jacobian-sized footprints (FOOT_*).
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
struct Params { D:f32, k:f32, contrast:f32, morph:f32, sigPre:f32, sigC:f32, sub:f32, nw:f32, beta:f32, p0:f32, p1:f32, p2:f32,
  Kxx:f32, Kxy:f32, Kyy:f32, shx:f32, shy:f32, sdx:f32, sdy:f32, dix:f32, diy:f32, diz:f32, fadeW:f32, p3:f32 }
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
fn mult(x:u32, y:u32)->vec2f {
  if (MODE == 0u) { // Poisson: 1/(-lam*k), lam = -(4sin^2+4sin^2)*N^2 (5-point Laplacian eigenvalue), DC zeroed
    if (x == 0u && y == 0u) { return vec2f(0.0); }
    // operator: tr(K Hess h) + (s/D).grad h  ->  symbol -Q N^2 + i b ;  h^ = rhs^ / (Q N^2 - i b)
    let sx = sin(2.0 * PI * frq(x) / f32(N)); let sy = sin(2.0 * PI * frq(y) / f32(N));
    let Q = max(prm.Kxx * lamS(x) + prm.Kyy * lamS(y) + 2.0 * prm.Kxy * sx * sy, 1.0e-12);
    let a = Q * f32(N * N); let b = f32(N) * (prm.sdx * sx + prm.sdy * sy);
    return vec2f(a, b) / (a * a + b * b);
  }
  let s = select(prm.sigC, prm.sigPre, MODE == 1u);
  let kx = 2.0 * PI * frq(x) / f32(N); let ky = 2.0 * PI * frq(y) / f32(N);
  return vec2f(exp(-0.5 * s * s * (kx*kx + ky*ky)), 0.0);
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
  for (var i = lid; i < N; i += WGS) { sh[s + i] = cmul(sh[s + i], mult(c, i)); }
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
${bilFn('bilT', 'Tb')}
@compute @workgroup_size(8,8) fn initT(@builtin(global_invocation_id) g:vec3u) {
  let i = g.y * N + g.x;
  let fx = f32(g.x) + f32(N) * prm.shx; let fy = f32(g.y) + f32(N) * prm.shy;
  let fx0 = floor(fx); let fy0 = floor(fy);
  cbuf[i] = vec2f(bilT(i32(fx0) & (NI - 1), i32(fy0) & (NI - 1), fx - fx0, fy - fy0) - 1.0, 0.0);
}
// Monge-Ampere backward step (oblique): land position x + sDes + (s/D) h + K grad h, sample T there, write RHS (1 - 1/T_warped)
@compute @workgroup_size(8,8) fn maWarp(@builtin(global_invocation_id) g:vec3u) {
  let x = i32(g.x); let y = i32(g.y); let i = g.y * N + g.x;
  let xp = (x + 1) & (NI - 1); let xm = (x + NI - 1) & (NI - 1); let yp = (y + 1) & (NI - 1); let ym = (y + NI - 1) & (NI - 1);
  let s = f32(N) * 0.5;
  let hx = (field[u32(y * NI + xp)] - field[u32(y * NI + xm)]) * s;
  let hy = (field[u32(yp * NI + x)] - field[u32(ym * NI + x)]) * s;
  let hh = field[i];
  let Lx = prm.shx + prm.Kxx * hx + prm.Kxy * hy + prm.sdx * hh; let Ly = prm.shy + prm.Kxy * hx + prm.Kyy * hy + prm.sdy * hh;
  let fx = f32(x) + f32(N) * Lx; let fy = f32(y) + f32(N) * Ly;
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
fn fadeMask(fx:f32, fy:f32)->f32 {
  if (prm.fadeW <= 0.0) { return 1.0; }
  let px = fx / f32(N); let py = fy / f32(N); let f = prm.fadeW;
  return smoothstep(0.0, f, px) * smoothstep(0.0, f, 1.0 - px) * smoothstep(0.0, f, py) * smoothstep(0.0, f, 1.0 - py);
}
// CausticTankClock: footprint splat. Rays stay on the regular texel grid (no jitter); each ray deposits an anisotropic gaussian sized by the
// local landing Jacobian J (columns = landing spacing of the x / y neighbours): per principal axis sd = max(FOOT_MIN, FOOT_SPREAD x spacing), normalised per ray.
// Where light spreads, footprints overlap their neighbours -> a continuous field instead of a dot lattice; where it focuses (digits), the
// footprint shrinks to FOOT_MIN ~ the old bilinear tent (sd 0.41), so edges keep their sharpness. Folds need no area division: no fireflies.
const FOOT_SPREAD: f32 = 0.6;   // gaussian sd in units of the local ray spacing (0.6: ripple of a regular grid ~0.2%)
const FOOT_MIN: f32 = 0.4;      // minimum sd (texels), used wherever light is focused: ~ the old bilinear tent (sd 0.41)
const FOOT_RMAX: i32 = 8;       // footprint half-extent cap (texels): bounds the cost at strongly diverging rays (truncated, still normalised)
// landing displacement (texels) of the ray leaving texel (x,y) (any integer, wrapped): exact vector Snell, oblique collimated light, floor at depth D
fn landD(x:i32, y:i32)->vec2f {
  let i = u32((y & (NI - 1)) * NI + (x & (NI - 1)));
  let gx = hxb[i]; let gy = hyb[i]; let hh = field[i]; let eta = 1.0 / prm.nw;
  let inv = 1.0 / sqrt(1.0 + gx*gx + gy*gy);
  let n = vec3f(-gx * inv, -gy * inv, inv);
  let di = vec3f(prm.dix, prm.diy, prm.diz);
  let c = -dot(di, n); let kk = 1.0 - eta*eta*(1.0 - c*c);
  let f = eta * c - sqrt(kk);
  let t = eta * di + f * n;
  let len = (prm.D + hh) / (-t.z);
  return vec2f(len * t.x, len * t.y) * f32(N);
}
// forward caustic: one ray per texel refracted to the floor, deposited with its Jacobian-sized gaussian footprint (periodic target, fadeMask at the landing point)
@compute @workgroup_size(8,8) fn splat(@builtin(global_invocation_id) g:vec3u) {
  let x = i32(g.x); let y = i32(g.y);
  let p = vec2f(f32(x), f32(y)) + landD(x, y);
  let w = fadeMask(p.x, p.y);
  if (!(abs(p.x) < 1.0e6 && abs(p.y) < 1.0e6) || w <= 0.0) { return; }
  let a = vec2f(1.0, 0.0) + (landD(x + 1, y) - landD(x - 1, y)) * 0.5;   // d(landing)/dx
  let b = vec2f(0.0, 1.0) + (landD(x, y + 1) - landD(x, y - 1)) * 0.5;   // d(landing)/dy
  // per principal direction of J J^T: sd = max(FOOT_MIN, FOOT_SPREAD * spacing) -> only spreading directions widen, focused ones stay FOOT_MIN
  let axx = a.x*a.x + b.x*b.x; let ayy = a.y*a.y + b.y*b.y; let axy = a.x*a.y + b.x*b.y;
  let hd = sqrt(0.25*(axx - ayy)*(axx - ayy) + axy*axy); let l1 = 0.5*(axx + ayy) + hd; let l2 = max(0.5*(axx + ayy) - hd, 0.0);
  var v1 = select(vec2f(0.0, 1.0), vec2f(1.0, 0.0), axx >= ayy);
  if (abs(axy) > 1e-9) { v1 = normalize(vec2f(axy, l1 - axx)); }
  let s2 = FOOT_SPREAD * FOOT_SPREAD; let m2 = FOOT_MIN * FOOT_MIN;
  let q1 = max(m2, s2 * l1); let q2 = max(m2, s2 * l2);
  let cxx = q1*v1.x*v1.x + q2*v1.y*v1.y; let cyy = q1*v1.y*v1.y + q2*v1.x*v1.x; let cxy = (q1 - q2)*v1.x*v1.y;
  let det = cxx * cyy - cxy * cxy;
  let ixx = cyy / det; let iyy = cxx / det; let ixy = -cxy / det;
  let rx = min(i32(ceil(2.5 * sqrt(cxx))), FOOT_RMAX); let ry = min(i32(ceil(2.5 * sqrt(cyy))), FOOT_RMAX);
  let X0 = i32(floor(p.x)); let Y0 = i32(floor(p.y));
  var sum = 0.0;
  for (var j = -ry; j <= ry + 1; j++) { for (var i = -rx; i <= rx + 1; i++) {
    let d = vec2f(f32(X0 + i), f32(Y0 + j)) - p;
    sum += exp(-0.5 * (ixx*d.x*d.x + 2.0*ixy*d.x*d.y + iyy*d.y*d.y));
  } }
  let s = w * SCALE / max(sum, 1e-20);
  for (var j = -ry; j <= ry + 1; j++) { for (var i = -rx; i <= rx + 1; i++) {
    let d = vec2f(f32(X0 + i), f32(Y0 + j)) - p;
    let v = s * exp(-0.5 * (ixx*d.x*d.x + 2.0*ixy*d.x*d.y + iyy*d.y*d.y));
    if (v >= 0.5) { atomicAdd(&accum[u32(((Y0 + j) & (NI - 1)) * NI + ((X0 + i) & (NI - 1)))], u32(v + 0.5)); }
  } }
}
`;
}
