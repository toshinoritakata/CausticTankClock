// Clock logic: displayed time (with a debug offset), layout/format settings and the glyph key.
// The glyph key is 'CLK|layout|line1|line2' (layout 'single' has one line); it doubles as the cache key of the raw glyph canvas.
const pad2 = n => String(n).padStart(2, '0');

export function createClock($) {
  // clk.off: debug time offset in ms (window.clock.setTime/advance); key/last: current vs last-solved glyph key; ticks: retarget count
  const clk = { off: 0, key: null, last: null, ticks: 0 };
  const clockOn = () => $('clockOn').checked;
  const clockDate = () => new Date(Date.now() + clk.off);
  function clockParts() {
    const d = clockDate(); let H = d.getHours(); if ($('clock12').value === '12') H = H % 12 || 12;
    return [pad2(H), pad2(d.getMinutes())];
  }
  const shownStr = () => clockParts().join('');   // displayed digits, e.g. '0905'
  function clockKey() {
    const [H, M] = clockParts();
    return $('clockLayout').value === 'single' ? 'CLK|single|' + H + M : 'CLK|stacked|' + H + '|' + M;
  }
  return { clk, clockOn, clockParts, clockKey, shownStr };
}
