// Float64 CPU reference of the oblique Monge-Ampere design + exact oblique forward caustic (mirrors solver.js; small N only).
import { NW } from './oblique.js';
const fftCache = {};
function plan(n) {
  if (fftCache[n]) return fftCache[n];
  const rev = new Uint32Array(n), bits = Math.log2(n);
  for (let i = 0; i < n; i++) { let r = 0; for (let b = 0; b < bits; b++) if (i & (1 << b)) r |= 1 << (bits - 1 - b); rev[i] = r; }
  const cs = new Float64Array(n / 2), sn = new Float64Array(n / 2);
  for (let i = 0; i < n / 2; i++) { cs[i] = Math.cos(2 * Math.PI * i / n); sn[i] = -Math.sin(2 * Math.PI * i / n); }
  return fftCache[n] = { rev, cs, sn, br: new Float64Array(n), bi: new Float64Array(n) };
}
function fft1(re, im, off, stride, n, inv, p) {
  const { rev, cs, sn, br, bi } = p;
  for (let i = 0; i < n; i++) { const j = rev[i]; br[j] = re[off + i * stride]; bi[j] = im[off + i * stride]; }
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1, step = n / size;
    for (let s = 0; s < n; s += size) for (let k = 0; k < half; k++) {
      const wr = cs[k * step], wi = inv ? -sn[k * step] : sn[k * step], a = s + k, b = a + half;
      const tr = br[b] * wr - bi[b] * wi, ti = br[b] * wi + bi[b] * wr;
      br[b] = br[a] - tr; bi[b] = bi[a] - ti; br[a] += tr; bi[a] += ti;
    }
  }
  const sc = inv ? 1 / n : 1;
  for (let i = 0; i < n; i++) { re[off + i * stride] = br[i] * sc; im[off + i * stride] = bi[i] * sc; }
}
function fft2(re, im, n, inv) { const p = plan(n); for (let y = 0; y < n; y++) fft1(re, im, y * n, 1, n, inv, p); for (let x = 0; x < n; x++) fft1(re, im, x, n, n, inv, p); }
const frq = (i, n) => (i <= n / 2 ? i : i - n);
export function blur(a, n, sigma) {
  const re = Float64Array.from(a), im = new Float64Array(n * n); fft2(re, im, n, false);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const kx = 2 * Math.PI * frq(x, n) / n, ky = 2 * Math.PI * frq(y, n) / n, g = Math.exp(-0.5 * sigma * sigma * (kx * kx + ky * ky));
    re[y * n + x] *= g; im[y * n + x] *= g;
  }
  fft2(re, im, n, true); return re;
}
const mean = a => { let s = 0; for (const v of a) s += v; return s / a.length; };
// h^ = rhs^ / (Q N^2 - i b),  Q = Kxx*4sin^2(pi x/N) + Kyy*4sin^2(pi y/N) + 2Kxy sin(th_x) sin(th_y),  b = N (sdx sin th_x + sdy sin th_y)
function solve(rhs, n, L) {
  const re = Float64Array.from(rhs), im = new Float64Array(n * n); fft2(re, im, n, false);
  const [Kxx, Kxy, Kyy] = L.K, [sdx, sdy] = L.sOverD;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const i = y * n + x;
    if (x === 0 && y === 0) { re[i] = 0; im[i] = 0; continue; }
    const sx = Math.sin(2 * Math.PI * frq(x, n) / n), sy = Math.sin(2 * Math.PI * frq(y, n) / n);
    const Q = Math.max(Kxx * 4 * Math.sin(Math.PI * x / n) ** 2 + Kyy * 4 * Math.sin(Math.PI * y / n) ** 2 + 2 * Kxy * sx * sy, 1e-12);
    const a = Q * n * n, b = n * (sdx * sx + sdy * sy), d = a * a + b * b, mr = a / d, mi = b / d;
    const r = re[i], q = im[i]; re[i] = r * mr - q * mi; im[i] = r * mi + q * mr;
  }
  fft2(re, im, n, true); return re;
}
function grad(h, n) {
  const hx = new Float64Array(n * n), hy = new Float64Array(n * n), s = n / 2;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const i = y * n + x, xp = (x + 1) % n, xm = (x + n - 1) % n, yp = (y + 1) % n, ym = (y + n - 1) % n;
    hx[i] = (h[y * n + xp] - h[y * n + xm]) * s; hy[i] = (h[yp * n + x] - h[ym * n + x]) * s;
  }
  return { hx, hy };
}
function bilP(T, n, fx, fy) { // periodic bilinear
  const x0 = Math.floor(fx), y0 = Math.floor(fy), ax = fx - x0, ay = fy - y0;
  const X0 = ((x0 % n) + n) % n, X1 = (X0 + 1) % n, Y0 = ((y0 % n) + n) % n, Y1 = (Y0 + 1) % n;
  return (T[Y0 * n + X0] * (1 - ax) + T[Y0 * n + X1] * ax) * (1 - ay) + (T[Y1 * n + X0] * (1 - ax) + T[Y1 * n + X1] * ax) * ay;
}
const sstep = (e0, e1, x) => { const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t); };
const fadeMask = (px, py, f) => f <= 0 ? 1 : sstep(0, f, px) * sstep(0, f, 1 - px) * sstep(0, f, py) * sstep(0, f, 1 - py);

export function targetT(grayA, grayB, n, p) {
  const g = new Float64Array(n * n); for (let i = 0; i < n * n; i++) g[i] = grayA[i] + (grayB[i] - grayA[i]) * p.morph;
  const f = blur(g, n, p.preblur), m = mean(f) + 1e-9, T = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) T[i] = Math.max(0.05, 1 + p.contrast * (f[i] / m - 1));
  const m2 = mean(T); for (let i = 0; i < n * n; i++) T[i] /= m2; return T;
}
export function designH(T, n, p, L) {
  const [Kxx, Kxy, Kyy] = L.K, [sdx, sdy] = L.sOverD, [shx, shy] = L.shDes, rhs = new Float64Array(n * n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) rhs[y * n + x] = bilP(T, n, x + n * shx, y + n * shy) - 1;
  let h = solve(rhs, n, L);
  for (let it = 0; it < p.iters; it++) {
    const { hx, hy } = grad(h, n);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const i = y * n + x, hh = h[i];
      const Lx = shx + Kxx * hx[i] + Kxy * hy[i] + sdx * hh, Ly = shy + Kxy * hx[i] + Kyy * hy[i] + sdy * hh;
      rhs[i] = 1 - 1 / bilP(T, n, x + n * Lx, y + n * Ly);
    }
    const hn = solve(rhs, n, L); for (let i = 0; i < n * n; i++) h[i] += p.beta * (hn[i] - h[i]);
  }
  return h;
}
export function forward(h, n, D, L, sigC) {
  const { hx, hy } = grad(h, n), out = new Float64Array(n * n), eta = 1 / NW, di = L.di;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const i = y * n + x, inv = 1 / Math.sqrt(1 + hx[i] * hx[i] + hy[i] * hy[i]), nn = [-hx[i] * inv, -hy[i] * inv, inv];
    const c = -(di[0] * nn[0] + di[1] * nn[1] + di[2] * nn[2]), f = eta * c - Math.sqrt(1 - eta * eta * (1 - c * c));
    const t = [eta * di[0] + f * nn[0], eta * di[1] + f * nn[1], eta * di[2] + f * nn[2]], len = (D + h[i]) / -t[2];
    const fx = x + len * t[0] * n, fy = y + len * t[1] * n, w = fadeMask(fx / n, fy / n, L.fadeW);
    const x0 = Math.floor(fx), y0 = Math.floor(fy), ax = fx - x0, ay = fy - y0;
    const X0 = ((x0 % n) + n) % n, X1 = (X0 + 1) % n, Y0 = ((y0 % n) + n) % n, Y1 = (Y0 + 1) % n;
    out[Y0 * n + X0] += w * (1 - ax) * (1 - ay); out[Y0 * n + X1] += w * ax * (1 - ay); out[Y1 * n + X0] += w * (1 - ax) * ay; out[Y1 * n + X1] += w * ax * ay;
  }
  return sigC > 0 ? blur(out, n, sigC) : out;
}
export function cpuRun(grayA, grayB, n, p, L) {
  const T = targetT(grayA, grayB, n, p), h = designH(T, n, p, L), C = forward(h, n, p.D, L, p.blur);
  return { T, h, C };
}
