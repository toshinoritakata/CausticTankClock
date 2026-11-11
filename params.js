// Saved parameters (localStorage): auto-save of changed settings, restore at startup, reset, JSON export / import.
// Saved: every id'd range/checkbox/text/select in #ui + camera targets (yaw/pitch/dist) + look-at offset (via its slider).
// Not saved: clock time offset, panel open/fullscreen state, running state, HUD text.
// Format: { version: 1, params?: { id: value }, camera?: { ty, tp, td } }, stored as a diff vs. the page defaults.
const PKEY = 'CausticTankClock.params.v1', PVER = 1;
const SAVE_DELAY_MS = 300, POLL_MS = 1000, MAX_TEXT_LEN = 200;

// deps: $, qp, canvas, app; cam + readTargetSliders + snapCam (camera.js); camRange = { PITCH_MIN, PITCH_MAX, DIST_MIN, DIST_MAX };
// onImported(): refresh UI that depends on imported values (mesh select, compass)
export function initParams({ $, qp, canvas, app, cam, readTargetSliders, snapCam, camRange, onImported }) {
  const pctl = [...document.querySelectorAll('#ui input[id], #ui select[id]')].filter(el => ['range', 'checkbox', 'text'].includes(el.type) || el.tagName === 'SELECT');
  const valueOf = el => el.type === 'checkbox' ? el.checked : (el.type === 'range' ? +el.value : el.value);
  const pdef = {}; for (const el of pctl) pdef[el.id] = valueOf(el);
  const pdefCam = { ty: cam.ty, tp: cam.tp, td: cam.td }; app.paramDefaults = { params: pdef, camera: pdefCam };
  const psStatus = $('psStatus'), say = t => { psStatus.textContent = t; };
  const stamp = () => new Date().toTimeString().slice(0, 8);
  const lsGet = () => { try { return localStorage.getItem(PKEY); } catch { return null; } };
  const lsSet = v => { try { localStorage.setItem(PKEY, v); return true; } catch { return false; } };
  const lsDel = () => { try { localStorage.removeItem(PKEY); } catch { /* storage unavailable */ } };
  const collect = () => {
    const params = {}; for (const el of pctl) params[el.id] = valueOf(el);
    return { version: PVER, params, camera: { ty: +((((cam.ty % 360) + 360) % 360).toFixed(2)), tp: +cam.tp.toFixed(2), td: +cam.td.toFixed(3) } };
  };
  // diff vs. the page defaults: only intentional changes are stored, so later default changes still reach untouched settings
  const diffOf = full => {
    const params = {}; for (const [id, v] of Object.entries(full.params)) { const d = pdef[id]; if (typeof v === 'number' ? Math.abs(v - d) > 1e-9 : v !== d) params[id] = v; }
    const camera = {}, c = full.camera, wrap = x => ((x % 360) + 540) % 360 - 180;
    if (!full.params.orbit && Math.abs(wrap(c.ty - pdefCam.ty)) > 1e-3) camera.ty = c.ty;   // with auto-orbit on, yaw is never stored
    if (Math.abs(c.tp - pdefCam.tp) > 1e-3) camera.tp = c.tp;
    if (Math.abs(c.td - pdefCam.td) > 1e-3) camera.td = c.td;
    const o = { version: PVER }; if (Object.keys(params).length) o.params = params; if (Object.keys(camera).length) o.camera = camera;
    return Object.keys(o).length > 1 ? o : null;
  };
  const storeDiff = () => { const d = diffOf(collect()); if (!d) { lsDel(); return 'nothing to save (all defaults)'; } return lsSet(JSON.stringify(d)) ? null : 'save failed (storage unavailable)'; };

  // validate + apply; returns { ok, bad } counts. fire: dispatch input/change so live handlers run (import); at startup the normal startup path reads the values.
  function applyParams(o, fire) {
    if (!o || typeof o !== 'object' || o.version !== PVER || (o.params !== undefined && (typeof o.params !== 'object' || !o.params))) throw new Error('bad format / version');
    let ok = 0, bad = 0;
    for (const [id, v] of Object.entries(o.params || {})) {
      const el = pctl.find(e => e.id === id); if (!el) { bad++; continue; }
      let good = false;
      if (el.type === 'checkbox') { if (typeof v === 'boolean') { el.checked = v; good = true; } }
      else if (el.type === 'range') { if (typeof v === 'number' && isFinite(v) && v >= +el.min - 1e-9 && v <= +el.max + 1e-9) { el.value = v; good = true; } }
      else if (el.tagName === 'SELECT') { const opt = [...el.options].find(x => x.value === String(v)); if (opt) { el.value = opt.value; good = true; } }
      else if (typeof v === 'string' && v.length <= MAX_TEXT_LEN) { el.value = v; good = true; }
      if (!good) { bad++; continue; } ok++;
      if (fire) { el.dispatchEvent(new Event('input', { bubbles: true })); el.dispatchEvent(new Event('change', { bubbles: true })); }
    }
    const c = o.camera;
    if (c && typeof c === 'object') {
      const inRange = (x, lo, hi) => typeof x === 'number' && isFinite(x) && x >= lo && x <= hi;
      if (c.ty !== undefined && $('orbit').checked && !fire) { /* auto-orbit on: yaw is not restored */ } else if (inRange(c.ty, -1e6, 1e6)) { cam.ty = c.ty; ok++; } else if (c.ty !== undefined) bad++;
      if (inRange(c.tp, camRange.PITCH_MIN, camRange.PITCH_MAX)) { cam.tp = c.tp; ok++; } else if (c.tp !== undefined) bad++;
      if (inRange(c.td, camRange.DIST_MIN, camRange.DIST_MAX)) { cam.td = c.td; ok++; } else if (c.td !== undefined) bad++;
    }
    readTargetSliders(); if (!fire) snapCam();
    return { ok, bad };
  }

  // ----- auto-save -----
  let noSave = false, saveTimer = 0, lastJson = '';
  const writeNow = (label = 'saved') => { if (noSave) return; lastJson = JSON.stringify(collect().params); const err = storeDiff(); say(err || `${label} ${stamp()}`); };
  const saveSoon = () => { clearTimeout(saveTimer); saveTimer = setTimeout(() => writeNow('auto-saved'), SAVE_DELAY_MS); };
  const ui = $('ui'); ui.addEventListener('input', saveSoon); ui.addEventListener('change', saveSoon);
  canvas.addEventListener('pointerup', saveSoon); canvas.addEventListener('wheel', saveSoon, { passive: true });
  document.querySelectorAll('button[data-p]').forEach(b => b.addEventListener('click', saveSoon));
  window.addEventListener('keydown', e => { if (!(e.target.tagName === 'INPUT' && e.target.type === 'text')) saveSoon(); });   // keyboard toggles (h/g/o/c/1-3) fire no events
  setInterval(() => { if (JSON.stringify(collect().params) !== lastJson) saveSoon(); }, POLL_MS);
  const flush = () => { if (!noSave) { clearTimeout(saveTimer); try { storeDiff(); } catch { /* ignore */ } } };
  window.addEventListener('pagehide', flush); document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'hidden') flush(); });

  // ----- buttons: save / reset / export / import -----
  $('psSave').onclick = () => { clearTimeout(saveTimer); writeNow('saved'); };
  $('psReset').onclick = () => { noSave = true; clearTimeout(saveTimer); lsDel(); const u = new URL(location.href); u.searchParams.delete('reset'); location.replace(u.toString()); };
  $('psExport').onclick = () => {
    const blob = new Blob([JSON.stringify(collect(), null, 2)], { type: 'application/json' }), a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = 'CausticTankClock-params.json'; document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 1000); say('exported ' + stamp());
  };
  const fileIn = document.createElement('input'); fileIn.type = 'file'; fileIn.accept = 'application/json,.json'; fileIn.style.display = 'none'; document.body.appendChild(fileIn);
  app.importParams = text => { const r = applyParams(JSON.parse(text), true); onImported(); clearTimeout(saveTimer); writeNow('imported+saved'); say(psStatus.textContent + ` (${r.ok} ok, ${r.bad} ignored)`); return r; };
  fileIn.onchange = async () => { const f = fileIn.files[0]; fileIn.value = ''; if (!f) return; try { app.importParams(await f.text()); } catch (e) { say('import failed: ' + e.message); } };
  $('psImport').onclick = () => fileIn.click();
  document.querySelectorAll('#psBtns button').forEach(b => b.addEventListener('click', () => b.blur()));

  // ----- startup: ?reset=1 clears the store, otherwise restore it (falling back to the defaults if invalid) -----
  if (qp.get('reset') === '1') { lsDel(); say('stored params cleared (?reset=1)'); }
  else {
    const raw = lsGet();
    if (raw !== null) {
      try { const r = applyParams(JSON.parse(raw), false); say(`restored (${r.ok} ok, ${r.bad} ignored)`); }
      catch (e) { console.warn('param restore failed, using defaults', e); lsDel(); applyParams({ version: PVER, params: pdef, camera: pdefCam }, false); say('stored params invalid: reset to defaults'); }
    }
  }
  lastJson = JSON.stringify(collect().params);
}
