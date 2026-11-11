// Wave dynamics parameters, time-step stability and the continuous ambient forcing. rd(id) reads a UI value as a number.
const clamp01 = x => Math.min(1, Math.max(0, x));
const TWO_PI = 2 * Math.PI;
const SECONDS_STIFFEN = [1.8, 1.6];   // clock shows seconds: the surface is retargeted every second, so kappa x1.8 and gamma x1.6 make it settle quickly
const STABILITY_LIMIT = 3.2;          // 80 % of the symplectic-Euler limit (4), see dtSafe

// solver/wave parameters from the UI; kmax is clipped to the grid (N/8 cycles)
export function dynParams(rd, N, secondsShown) {
  const q = secondsShown ? SECONDS_STIFFEN : [1, 1];
  return { c2: rd('c') ** 2, kappa: rd('kappa') * q[0], gamma: rd('gamma') * q[1], nu: rd('nu') * 1e-5, s: rd('s') * 1e-7, kmax: Math.min(rd('kmax'), N / 8), cap: rd('sub') };
}

// Symplectic Euler with explicit friction, per Fourier mode: stable iff dt^2 w^2 < 4 - 2 dt G  (w^2 = omega^2, G = gamma + nu*lambda, lambda = 8N^2 = 5-pt Laplacian max).
// omega^2_max = c^2*8N^2 + s*64N^4 + kappa (5-/13-point stencils, h = 1/N). We take 80 % of the limit: w^2 dt^2 + 2 G dt = 3.2  ->  dt = (-G + sqrt(G^2 + 3.2 w^2)) / w^2.
export function dtSafe(P, N) {
  const w2 = P.c2 * 8 * N * N + P.s * 64 * N ** 4 + P.kappa, G = P.gamma + P.nu * 8 * N * N;
  return (-G + Math.sqrt(G * G + STABILITY_LIMIT * w2)) / w2;
}

// kappa ramp-up after (re)start: smoothstep over 'ramp' seconds (0 = off)
export function kappaEff(P, t, ramp) { if (ramp <= 0) return P.kappa; const q = clamp01(t / ramp); return P.kappa * q * q * (3 - 2 * q); }

// substeps for a frame of dt seconds: n capped at P.cap (then 'slowed' if the step would exceed the stable one)
export function planSteps(dt, P, N) {
  const ds = dtSafe(P, N); let n = Math.max(1, Math.ceil(dt / ds)), slowed = false;
  if (n > P.cap) { n = P.cap; slowed = dt / n > ds; }
  return { n, h: Math.min(dt / n, ds), slowed, ds };
}

// ---------- continuous ambient forcing: e_tt += F(x,t), F = sum_m A_m mod_m(t) cos(2pi k_m.x - phi_m(t)) ----------
// 12 integer wave vectors (periodic) around |k| ~ 'ambient scale' cycles; frequency Omega_m = speed * omega0(k_m) (dispersion relation of the model, speed<1 = off-resonant);
// slow amplitude modulation (10-30 s) with random phases; phases advance with simulation time. The gain is normalised analytically with the steady-state transfer
// |H| = 1/sqrt((w0^2-W^2)^2 + (G W)^2), G = gamma + nu k^2, so that rms(e) = ambient * rms(h*) whatever N / kappa / gamma / nu / c / s are.
const AMB_MODES = 12, AMB_SEED = 4242, AMB_MOD_DEPTH = 0.6;
export function createAmbient({ device, $, rd, mulberry32 }) {
  const AMB = { modes: [], scaleKey: null, w: new Float32Array(AMB_MODES * 4), simT: 0 };
  function ambModes(scale) {
    const rnd = mulberry32(AMB_SEED), out = []; let guard = 0;
    while (out.length < AMB_MODES && guard++ < 5000) {
      const kmin = Math.max(1.5, scale * 0.5), kmax = scale * 1.5 + 1;
      const kx = Math.round((rnd() * 2 - 1) * kmax), ky = Math.round((rnd() * 2 - 1) * kmax), kk = Math.hypot(kx, ky);
      if (kk < kmin || kk > kmax) continue;
      if (out.some(o => (o.kx === kx && o.ky === ky) || (o.kx === -kx && o.ky === -ky))) continue;
      out.push({ kx, ky, kk, ph: rnd() * TWO_PI, T: 10 + rnd() * 20, psi: rnd() * TWO_PI, dir: rnd() < 0.5 ? 1 : -1, w: 1 / Math.sqrt(kk), phi: rnd() * TWO_PI });
    }
    return out;
  }
  // write this frame's forcing table to c.ambU; dt = simulated seconds, rmsStar = rms(h*)
  function update(c, dt, P, rmsStar) {
    const on = $('ambOn').checked, r = rd('ambient'), sc = Math.round(rd('ascale') * 4) / 4, w = AMB.w; w.fill(0);
    if (AMB.scaleKey !== sc) { AMB.modes = ambModes(sc); AMB.scaleKey = sc; }
    if (!on || r <= 0 || !(rmsStar > 0)) { device.queue.writeBuffer(c.ambU, 0, w); return; }
    const M = AMB.modes, sp = rd('aspeed'); AMB.simT += dt;
    let sw2 = 0; for (const m of M) sw2 += m.w * m.w;   // E_m = g w_m; sum E_m^2 <mod^2>/2 = target^2
    const g = r * rmsStar / Math.sqrt(sw2 * (1 + AMB_MOD_DEPTH * AMB_MOD_DEPTH / 2) / 2);
    M.forEach((m, i) => {
      const k = TWO_PI * m.kk, w0sq = P.c2 * k * k + P.kappa + P.s * k ** 4, w0 = Math.sqrt(w0sq), W = sp * w0, G = P.gamma + P.nu * k * k;
      const D = Math.sqrt((w0sq - W * W) ** 2 + (G * W) ** 2);   // forcing amplitude per unit e amplitude
      m.phi += m.dir * W * dt;
      const mod = 1 + AMB_MOD_DEPTH * Math.sin(TWO_PI * AMB.simT / m.T + m.psi);
      w[4 * i] = TWO_PI * m.kx; w[4 * i + 1] = TWO_PI * m.ky; w[4 * i + 2] = m.phi + m.ph; w[4 * i + 3] = g * m.w * D * mod;
    });
    device.queue.writeBuffer(c.ambU, 0, w);
  }
  const state = () => ({ simT: AMB.simT, n: AMB.modes.length, modes: AMB.modes.map(m => [m.kx, m.ky, +m.T.toFixed(1)]) });
  return { update, state };
}
