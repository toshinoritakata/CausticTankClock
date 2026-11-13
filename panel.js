// Panel element lookup that tolerates a stale or edited index.html (e.g. a cached page after a deploy): the modules are cache-busted with ?v=, the
// page itself is not, so new code can ask for a control the old HTML lacks. A missing element is replaced by a detached stand-in holding the
// default below (one console warning per id), so the parameter takes its default and its wiring is a harmless no-op instead of a crash.
// PANEL_DEFAULTS mirrors the value / checked / selected defaults in index.html; tank.checkPanelDefaults() lists any drift between the two.
export const PANEL_DEFAULTS = { clockOn: true, trans: "1.5", kick: "3", ambOn: true, ambient: "0.02", ascale: "4", aspeed: "0.85", ripple: "0.003", rscale: "14", textList: "A,S,R", hold: "4", thr: "0.04", hudOn: true, elev: "65", az: "30", center: true, cast: "0.5", fit: true, wrap: true, c: "0.15", kappa: "3", gamma: "0.8", nu: "40", s: "4", sub: "32", amp: "0.6", kmax: "7", seed: "7", det: true, ramp: "0", glow: "0", D: "0.9", contrast: "0.7", iters: "8", expo: "1", relief: "1", decouple: false, rblur: "1", disp: "1", flat: false, glass: true, orbit: true, ospeed: "7", god: true, rgain: "1", rsig: "0.8", rsteps: "32", rg: "0.5", rnoise: "0.3", nz: "24", tgy: "0", clockLayout: "stacked", clock12: "24", N: "256", mesh: "512", csamp: "bilinear", volres: "256" };
const STANDIN_TAG = { compass: 'canvas' };

export function makeLookup(doc = document) {
  const standins = new Map();
  return id => {
    const el = doc.getElementById(id); if (el) return el;
    let s = standins.get(id); if (s) return s;
    const def = PANEL_DEFAULTS[id];
    if (typeof def === 'boolean') { s = doc.createElement('input'); s.type = 'checkbox'; s.checked = def; }
    else if (def !== undefined) { s = doc.createElement('input'); s.value = def; }
    else s = doc.createElement(STANDIN_TAG[id] || 'div');
    console.warn(`panel: #${id} is missing from index.html (stale page?), using ${def !== undefined ? 'default ' + JSON.stringify(def) : 'a detached stand-in'}`);
    standins.set(id, s); return s;
  };
}

// ids whose HTML default differs from PANEL_DEFAULTS, or that are absent from the page ([] = in sync)
export function checkDefaults(doc = document) {
  const out = [];
  for (const [id, def] of Object.entries(PANEL_DEFAULTS)) {
    const el = doc.getElementById(id); if (!el) { out.push([id, 'missing']); continue; }
    const html = el.type === 'checkbox' ? el.defaultChecked : el.tagName === 'SELECT' ? ([...el.options].find(o => o.defaultSelected) || el.options[0]).value : el.defaultValue;
    if (html !== def) out.push([id, html, def]);
  }
  return out;
}
