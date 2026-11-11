// Glyph texture: draw a letter / the clock key into a 2-D canvas, then warp it into the floor-plane cast shadow.
// Pure canvas code; the caller (app.js uploadGlyph) copies the result to the solver's input texture.
const RAD = Math.PI / 180;

// Clock key 'CLK|layout|line1|line2|line3' (see clock.js clockKey): lines are fitted into the square and centred.
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

export function drawGlyph(cv, text, N) {
  const c = cv.getContext('2d', { willReadFrequently: true }); cv.width = cv.height = N;
  c.fillStyle = '#000'; c.fillRect(0, 0, N, N); c.fillStyle = '#fff';
  if (text.startsWith('CLK|')) { drawClockGlyph(c, text, N); return; }
  let fs = N * 0.85; const fam = '"Arial Black", Helvetica, sans-serif';
  c.font = `900 ${fs}px ${fam}`;
  const wdt = c.measureText(text).width; if (wdt > N * 0.92) { fs *= N * 0.92 / wdt; c.font = `900 ${fs}px ${fam}`; }
  c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillText(text, N / 2, N * 0.54);
}

// Cast-shadow warp of the glyph (floor plane, about the floor centre): stretch sa = 1 + cast*(1/sin(el) - 1) along the azimuth ('sin' law = beam-footprint stretch;
// stretchLaw 'tan' = upright-object shadow length), 1 perpendicular; optional uniform shrink so the warped glyph bbox fits inside ~0.8 of the floor (scale-to-fit).
// Reads c.raw[0], writes c.cv[0]; q = { cast, el, az, fit }.
export function castWarp(c, q, stretchLaw) {
  const N = c.N, cast = q.cast || 0; c.warpInfo = { sa: 1, f: 1 };
  for (const cv of c.cv) cv.width = cv.height = N;
  if (cast <= 0) { c.cv[0].getContext('2d').drawImage(c.raw[0], 0, 0); return; }
  const el = Math.min(90, Math.max(30, q.el)), a = q.az * RAD;
  const sa = Math.min(3, Math.max(0.3, 1 + cast * ((stretchLaw === 'tan' ? 1 / Math.tan(el * RAD) : 1 / Math.sin(el * RAD)) - 1)));
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
