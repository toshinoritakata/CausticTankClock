// Panel + HUD wiring: throttled re-solve on slider input, compass, buttons, fullscreen / kiosk, keyboard shortcuts, HUD text.
// deps: $, rd, qp, R (sim state), clock = createClock(), cam/app, and callbacks { solveStar, setN, initSim, nextLetter, disturb, curLetter }.
const RAD = Math.PI / 180;
const RESOLVE_MIN_INTERVAL_MS = 120, RESOLVE_BUSY_RETRY_MS = 40;
const HUD_BAR_LEN = 30;
// controls that change the designed surface h* (re-solved live while dragging)
const RESOLVE_IDS = ['D', 'contrast', 'iters', 'rblur', 'disp', 'elev', 'az', 'center', 'wrap', 'cast', 'fit'];
// range controls whose value is echoed in the span id + 'v'
const VALUE_LABEL_IDS = ['c', 'kappa', 'gamma', 'nu', 's', 'sub', 'amp', 'kmax', 'seed', 'hold', 'thr', 'ramp', 'glow', 'D', 'contrast', 'iters', 'expo', 'relief', 'rgain', 'rsig', 'rsteps', 'rg', 'rnoise', 'nz', 'elev', 'az', 'cast', 'rblur', 'disp', 'kick', 'trans', 'ascale', 'aspeed', 'ambient', 'ripple', 'rscale', 'ospeed', 'tgy'];

export function initUI({ $, rd, qp, R, clock, app, solveStar, setN, initSim, nextLetter, disturb, curLetter }) {
  const { clk, clockOn, clockKey, shownStr } = clock;

  // throttled re-solve (leading edge after >=120 ms since the last one, trailing edge guaranteed): live slider dragging keeps retargeting e smoothly
  let solveTimer = 0, lastSolve = 0;
  const resolveSoon = () => {
    clearTimeout(solveTimer);
    const run = () => { if (R.busy) { solveTimer = setTimeout(run, RESOLVE_BUSY_RETRY_MS); return; } lastSolve = performance.now(); solveStar(true); };
    solveTimer = setTimeout(run, Math.max(0, RESOLVE_MIN_INTERVAL_MS - (performance.now() - lastSolve)));
  };
  for (const id of RESOLVE_IDS) $(id).addEventListener('input', resolveSoon);

  // top view of the light direction: arrow = direction the light travels
  const cmp = $('compass').getContext('2d');
  function drawCompass() {
    const el = rd('elev'), az = rd('az') * RAD, r = 14, len = r * Math.cos(el * RAD) + (el >= 90 ? 0 : 3);
    cmp.clearRect(0, 0, 40, 40); cmp.strokeStyle = '#556'; cmp.lineWidth = 1; cmp.strokeRect(6, 6, 28, 28);
    cmp.strokeStyle = '#ffd88a'; cmp.fillStyle = '#ffd88a'; cmp.lineWidth = 2;
    const ex = 20 + len * Math.cos(az), ey = 20 + len * Math.sin(az); cmp.beginPath(); cmp.moveTo(20, 20); cmp.lineTo(ex, ey); cmp.stroke();
    cmp.beginPath(); cmp.arc(ex, ey, 2.5, 0, 7); cmp.fill();
  }
  for (const id of ['elev', 'az']) $(id).addEventListener('input', drawCompass);
  drawCompass();

  $('N').addEventListener('change', () => setN(+$('N').value));
  const syncMesh = () => { $('mesh').disabled = !$('decouple').checked; };
  $('decouple').addEventListener('change', syncMesh); syncMesh();

  const setPlay = v => { R.playing = v; $('play').textContent = v ? 'Pause' : 'Play'; };
  $('play').onclick = () => setPlay(!R.playing);
  $('replay').onclick = async () => { await initSim(); setPlay(true); };
  $('next').onclick = () => { if (!R.busy) nextLetter(); };
  $('disturb').onclick = () => { if (!R.busy) disturb(); };
  document.querySelectorAll('#ui button').forEach(b => b.addEventListener('click', () => b.blur()));

  const toggleFull = () => {
    if (document.body.classList.contains('full')) { document.fullscreenElement && document.exitFullscreen(); document.body.classList.remove('full'); }
    else { document.body.classList.add('full'); document.documentElement.requestFullscreen?.().catch(() => {}); }
  };
  $('fs').onclick = toggleFull;
  document.addEventListener('fullscreenchange', () => { if (!document.fullscreenElement) document.body.classList.remove('full'); });
  if (qp.get('full') === '1') document.body.classList.add('full', 'kiosk');   // exhibition: start with the panel hidden (pair with Chrome --kiosk)
  document.querySelectorAll('button[data-p]').forEach(b => b.onclick = () => app.preset(+b.dataset.p));

  // keyboard: f fullscreen, Esc leave it, space play/pause, r replay, n next, d disturb, h HUD, c clock, g god rays, o orbit, 1-3 camera presets
  window.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' && e.target.type === 'text') return;
    const k = e.key.toLowerCase();
    if (k === 'f') toggleFull();
    else if (e.key === 'Escape') document.body.classList.remove('full');
    else if (e.key === ' ') { e.preventDefault(); $('play').click(); }
    else if (k === 'r') $('replay').click();
    else if (k === 'n') $('next').click();
    else if (k === 'd') $('disturb').click();
    else if (k === 'h') $('hudOn').checked = !$('hudOn').checked;
    else if (k === 'c') { $('clockOn').checked = !$('clockOn').checked; $('clockOn').dispatchEvent(new Event('change')); }
    else if (k === 'g') $('god').checked = !$('god').checked;
    else if (k === 'o') $('orbit').checked = !$('orbit').checked;
    else if (e.key >= '1' && e.key <= '3') app.preset(+e.key);
  });

  // clock <-> letter-list switching
  $('clockOn').addEventListener('change', () => { if (clockOn()) { clk.key = clockKey(); clk.last = null; } else { R.listStr = null; } });

  const hudEl = $('hud');
  function updateHud() {
    hudEl.style.display = $('hudOn').checked ? 'block' : 'none'; if (!$('hudOn').checked) return;
    const r = R.resid, bar = '#'.repeat(Math.max(0, Math.min(HUD_BAR_LEN, Math.round(HUD_BAR_LEN * Math.min(1, r))))).padEnd(HUD_BAR_LEN, '.');
    const st = R.holdT > 0 ? `settled, holding ${R.holdT.toFixed(1)} / ${rd('hold')} s` : 'relaxing';
    hudEl.textContent = `${clockOn() ? 'clock ' + shownStr() + ' (' + clk.ticks + ' ticks)' : 'letter ' + curLetter()}   t = ${R.tl.toFixed(1)} s   residual |e|/|h*| = ${r.toFixed(3)}  [${bar}]  ${st}` + (R.slowed ? '   (sim slowed: substep cap)' : '');
  }
  const updateValueLabels = () => { for (const k of VALUE_LABEL_IDS) $(k + 'v').textContent = $(k).value; };

  return { drawCompass, syncMesh, setPlay, updateHud, updateValueLabels };
}
