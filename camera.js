// Camera: state, presets, pointer/wheel input (orbit / right-drag or shift-drag = pan the look-at height / wheel = zoom),
// per-frame easing + auto-orbit, and the view/projection matrices. Column-major mat4, WebGPU depth 0..1.
export const RAD = Math.PI / 180;
export const FOV = 34 * RAD;
export const PITCH_MIN = 4, PITCH_MAX = 89.5, DIST_MIN = 1.2, DIST_MAX = 6;   // also the valid range of restored camera params
const PRESETS = { 1: { yaw: 0, pitch: 89.5, dist: 3.4 }, 2: { yaw: 22, pitch: 52, dist: 3.3 }, 3: { yaw: 8, pitch: 14, dist: 3.1 } };
const DRAG_SPEED = 0.3;          // deg per px
const ORBIT_RESUME_MS = 1500;    // auto-orbit resumes this long after a drag ends
const EASE_RATE = 9;             // 1/s: camera approaches its targets as 1 - exp(-dt * EASE_RATE)
const TARGET_Y_RANGE = 1;        // look-at height offset limit (world units)
const TARGET_Y_FACTOR = 0.42;    // default look-at height = -D * TARGET_Y_FACTOR

// ---------- small vec / mat4 helpers ----------
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

// eye position, look-at basis and view-projection for water depth D
export function cameraView(cam, D, asp) {
  const tgt = [0, -D * TARGET_Y_FACTOR + cam.oy, 0];
  const cp = Math.cos(cam.pitch * RAD), eye = [tgt[0] + cam.dist * cp * Math.sin(cam.yaw * RAD), tgt[1] + cam.dist * Math.sin(cam.pitch * RAD), tgt[2] + cam.dist * cp * Math.cos(cam.yaw * RAD)];
  const up = cam.pitch > 89 ? [0, 0, -1] : [0, 1, 0];
  const lk = lookAt(eye, tgt, up);
  return { eye, lk, vp: mul(perspective(FOV, asp, 0.05, 60), lk.m) };
}

// cam: yaw/pitch/dist = current, ty/tp/td = eased targets, vy/vp = unused velocities (kept for tank.cam compatibility),
// oy = look-at height offset (world units) added to the default target, toy = its eased target. x/z of the target are fixed at 0.
export function createCamera({ $, rd, canvas, app }) {
  const cam = { yaw: 22, pitch: 21, dist: 3.3, ty: 22, tp: 21, td: 3.3, vy: 0, vp: 0, oy: 0, toy: 0 };
  app.cam = cam;
  function preset(k) { const p = PRESETS[k];   // take the shortest way round in yaw
    cam.ty = p.yaw; cam.tp = p.pitch; cam.td = p.dist; cam.yaw = cam.ty + ((((cam.yaw - cam.ty) % 360) + 540) % 360 - 180); }
  app.preset = k => { preset(k); cam.vy = cam.vp = 0; };   // presets do not stop the auto-orbit
  const snapCam = app.snapCam = () => { cam.yaw = cam.ty; cam.pitch = cam.tp; cam.dist = cam.td; cam.oy = cam.toy; };
  const syncTargetSliders = () => { $('tgy').value = (+cam.toy).toFixed(2); };
  const readTargetSliders = () => { cam.toy = +$('tgy').value; };
  $('tgy').addEventListener('input', readTargetSliders);
  $('tgreset').onclick = () => { cam.toy = 0; syncTargetSliders(); $('tgy').dispatchEvent(new Event('change', { bubbles: true })); };

  let drag = null, dragEnd = -1e9;
  canvas.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey }; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener('pointermove', e => {
    if (!drag) return;
    if (e.buttons === 0) { drag = null; dragEnd = performance.now(); return; }
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
    if (drag.pan) {   // pan: vertical drag moves only the look-at height, scaled by distance
      const sc = 2 * cam.dist * Math.tan(FOV / 2) / Math.max(1, canvas.clientHeight);
      cam.toy = Math.min(TARGET_Y_RANGE, Math.max(-TARGET_Y_RANGE, cam.toy + dy * sc)); syncTargetSliders();
    } else { cam.ty -= dx * DRAG_SPEED; cam.tp = Math.min(PITCH_MAX, Math.max(PITCH_MIN, cam.tp + dy * DRAG_SPEED)); }
  });
  canvas.addEventListener('pointerup', () => { drag = null; dragEnd = performance.now(); });
  canvas.addEventListener('contextmenu', e => e.preventDefault());
  canvas.addEventListener('wheel', e => { e.preventDefault(); cam.td = Math.min(DIST_MAX, Math.max(DIST_MIN, cam.td * Math.exp(e.deltaY * 0.001))); }, { passive: false });

  const camEl = $('camv');
  // per-frame: auto-orbit (pauses during a drag), ease towards targets, update the readout. now = rAF time (ms), dt in s
  function step(now, dt) {
    if ($('orbit').checked && !drag && now - dragEnd > ORBIT_RESUME_MS) cam.ty += rd('ospeed') * dt;   // ospeed in deg/s
    const a = 1 - Math.exp(-dt * EASE_RATE);
    cam.yaw += (cam.ty - cam.yaw) * a; cam.pitch += (cam.tp - cam.pitch) * a; cam.dist += (cam.td - cam.dist) * a;
    cam.oy += (cam.toy - cam.oy) * a;
    camEl.textContent = `camera  yaw ${(((cam.yaw % 360) + 360) % 360).toFixed(1)}  pitch ${cam.pitch.toFixed(1)}  dist ${cam.dist.toFixed(2)}  target y ${cam.oy.toFixed(2)}`;
  }
  return { cam, step, snapCam, readTargetSliders };
}
