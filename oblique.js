// Oblique collimated light: exact vector-Snell landing map + its linearisation (shared by GPU params and CPU reference).
// Solver frame: x,y = floor plane (texel x, texel y), z = up. Domain width = 1, floor at depth D below the mean surface.
// Light travels along d_i = (cos(el)cos(az), cos(el)sin(az), -sin(el)) (azimuth = horizontal direction of travel, from +x towards +y).
export const NW = 1.33;
export const EL_MIN = 30;
const rad = Math.PI / 180;

// Exact landing offset (relative to the surface point's x,y) of a ray hitting a surface of slope g=(gx,gy),
// floor at depth D below the (local) surface point. Vector Snell: t = eta*d + (eta*c - sqrt(1-eta^2(1-c^2))) n, c = -d.n
export function snellLand(di, gx, gy, D, nw = NW) {
  const inv = 1 / Math.sqrt(1 + gx * gx + gy * gy), n = [-gx * inv, -gy * inv, inv], eta = 1 / nw;
  const c = -(di[0] * n[0] + di[1] * n[1] + di[2] * n[2]);
  const f = eta * c - Math.sqrt(1 - eta * eta * (1 - c * c));
  const t = [eta * di[0] + f * n[0], eta * di[1] + f * n[1], eta * di[2] + f * n[2]];
  const len = D / -t[2];
  return [len * t[0], len * t[1]];
}

// Linearised design map: landing(x) = x + s + (s/D) h(x) + K grad h(x)   (domain units)
//  s    = flat-surface shift = D tan(theta_t) along the azimuth
//  s/D*h = ray starts at height h above mean surface -> path (D+h)/cos: first-order term, kept (the forward pass uses D+h too)
//  K    = d landing / d slope at zero slope, symmetric, = R diag(kpar,kperp) R^T ; K = D(1-1/nw) I at el=90
export function lightParams(elev, azim, D, center = true, wrap = false, nw = NW) {
  const el = Math.min(90, Math.max(EL_MIN, elev)), az = azim * rad;
  const ce = el >= 90 ? 0 : Math.cos(el * rad), se = Math.sin(el * rad);
  const di = [ce * Math.cos(az), ce * Math.sin(az), -se];
  const s = snellLand(di, 0, 0, D, nw);
  const e = 1e-6, a = snellLand(di, e, 0, D, nw), b = snellLand(di, -e, 0, D, nw), c = snellLand(di, 0, e, D, nw), d = snellLand(di, 0, -e, D, nw);
  const Jxx = (a[0] - b[0]) / (2 * e), Jyx = (a[1] - b[1]) / (2 * e), Jxy = (c[0] - d[0]) / (2 * e), Jyy = (c[1] - d[1]) / (2 * e);
  const Kxx = Jxx, Kyy = Jyy, Kxy = 0.5 * (Jxy + Jyx), asym = Math.abs(Jxy - Jyx);
  const u = [Math.cos(az), Math.sin(az)], v = [-u[1], u[0]];
  const kpar = u[0] * (Kxx * u[0] + Kxy * u[1]) + u[1] * (Kxy * u[0] + Kyy * u[1]);
  const kperp = v[0] * (Kxx * v[0] + Kxy * v[1]) + v[1] * (Kxy * v[0] + Kyy * v[1]);
  const kcross = u[0] * (Kxx * v[0] + Kxy * v[1]) + u[1] * (Kxy * v[0] + Kyy * v[1]);
  const sinT = ce / nw, tanT = sinT / Math.sqrt(1 - sinT * sinT);
  return {
    el, az: azim, di, s, sOverD: [s[0] / D, s[1] / D], K: [Kxx, Kxy, Kyy], asym, kpar, kperp, kcross, tanThetaT: tanT,
    shDes: center ? s : [0, 0],                  // shift used when designing (center ON: design against the shifted landing -> letter stays centred)
    fadeW: (!wrap && el < 89.9) ? 0.03 : 0,      // forward pass: drop/fade rays landing outside the floor instead of wrapping
    gain: se,                                    // floor irradiance / beam irradiance (cos of incidence on a horizontal plane); NOT applied (mean(T)=1 normalisation)
    sunWorld: [-ce * Math.cos(az), se, -ce * Math.sin(az)], // toward the light, world coords (x, up, z) ; world z == solver y
  };
}
