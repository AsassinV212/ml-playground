// Shared helpers for topic pages: sharp canvases, a colormap, a small line chart.
// Plain script (no modules) so pages also work when opened straight from disk.

(function () {
  "use strict";

  // Resize a canvas's backing store to its CSS size times the device pixel ratio,
  // so drawings stay sharp on Retina screens. Afterwards, draw in CSS pixels.
  function fitCanvas(canvas) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h, dpr };
  }

  // Dark-to-light ramp for heatmaps. t in [0, 1] -> [r, g, b].
  const RAMP = [
    [11, 20, 48],
    [26, 54, 110],
    [33, 112, 150],
    [70, 168, 150],
    [168, 214, 150],
  ];
  function colormap(t) {
    t = Math.min(1, Math.max(0, t)) * (RAMP.length - 1);
    const i = Math.min(RAMP.length - 2, Math.floor(t));
    const f = t - i;
    const a = RAMP[i], b = RAMP[i + 1];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  }

  // Compact number formatting for readouts.
  function fmt(v) {
    if (!isFinite(v)) return String(v);
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
    return v.toFixed(a < 1 ? 4 : 3);
  }

  // Draw a simple line chart of `values` (index on x, value on y).
  function drawLineChart(canvas, values, opts) {
    opts = opts || {};
    const { ctx, w, h } = fitCanvas(canvas);
    const css = getComputedStyle(document.documentElement);
    const muted = css.getPropertyValue("--muted").trim();
    const border = css.getPropertyValue("--border").trim();
    const color = opts.color || css.getPropertyValue("--accent").trim();

    const pad = { l: 8, r: 8, t: 18, b: 18 };
    ctx.clearRect(0, 0, w, h);
    ctx.font = "11px " + css.getPropertyValue("--mono");
    ctx.fillStyle = muted;
    ctx.strokeStyle = border;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, h - pad.b + 0.5);
    ctx.lineTo(w - pad.r, h - pad.b + 0.5);
    ctx.stroke();

    const finite = values.filter(isFinite);
    if (finite.length === 0) {
      ctx.fillText(opts.empty || "no data yet", pad.l, h / 2);
      return;
    }
    const minV = Math.min.apply(null, finite);
    const maxV = Math.max.apply(null, finite);
    // Pad a flat range so a single point sits in the middle instead of dividing by zero.
    const flat = maxV - minV < 1e-12;
    const lo = flat ? minV - 0.5 : minV;
    const hi = flat ? maxV + 0.5 : maxV;
    const n = Math.max(values.length - 1, opts.minSteps || 10);
    const X = (i) => pad.l + (i / n) * (w - pad.l - pad.r);
    const Y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (h - pad.t - pad.b);

    ctx.textBaseline = "top";
    ctx.fillText((opts.label || "") + " max " + fmt(maxV), pad.l, 2);
    ctx.textBaseline = "bottom";
    ctx.fillText("min " + fmt(minV), pad.l, h - 2);
    ctx.textAlign = "right";
    ctx.fillText("step " + (values.length - 1), w - pad.r, h - 2);
    ctx.textAlign = "left";

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.beginPath();
    values.forEach((v, i) => {
      if (!isFinite(v)) return;
      if (i === 0) ctx.moveTo(X(i), Y(v));
      else ctx.lineTo(X(i), Y(v));
    });
    ctx.stroke();
  }

  window.ML = { fitCanvas, colormap, fmt, drawLineChart };
})();
