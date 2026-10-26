// Shared helpers for topic pages: sharp canvases, a colormap, a small line chart.
// Plain script (no modules) so pages also work when opened straight from disk.

(function () {
  "use strict";

  // Translate if assets/i18n.js is loaded, otherwise keep English.
  const t = (s, vars) => (window.I18N ? window.I18N.t(s, vars) : s);

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
      ctx.fillText(t(opts.empty || "no data yet"), pad.l, h / 2);
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
    ctx.fillText(t(opts.label || "") + " " + t("max") + " " + fmt(maxV), pad.l, 2);
    ctx.textBaseline = "bottom";
    ctx.fillText(t("min") + " " + fmt(minV), pad.l, h - 2);
    ctx.textAlign = "right";
    ctx.fillText(t("step") + " " + (values.length - 1), w - pad.r, h - 2);
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

  // Read a CSS custom property, e.g. cssVar("--accent").
  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  // Colors for class labels 0, 1, 2 (coral, blue, yellow).
  const CLASS_COLORS = ["#ff7a59", "#5fb4ff", "#ffd166"];
  const CLASS_RGB = [[255, 122, 89], [95, 180, 255], [255, 209, 102]];

  // Seeded random numbers (mulberry32), so "random data" is reproducible.
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Standard normal sample (Box-Muller) from a uniform generator.
  function randn(rand) {
    const u = 1 - rand(), v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // A 2D plot: maps the world box [x0, x1] x [y0, y1] onto a fitted canvas.
  function plot2d(canvas, box) {
    const v = fitCanvas(canvas);
    const [x0, x1, y0, y1] = box;
    v.box = box;
    v.sx = (x) => ((x - x0) / (x1 - x0)) * v.w;
    v.sy = (y) => (1 - (y - y0) / (y1 - y0)) * v.h;
    v.wx = (sx) => x0 + (sx / v.w) * (x1 - x0);
    v.wy = (sy) => y0 + (1 - sy / v.h) * (y1 - y0);
    return v;
  }

  // Faint grid every `step` world units, with tick labels if `labels` is true.
  function drawGrid(v, step, labels) {
    const { ctx, w, h } = v;
    const [x0, x1, y0, y1] = v.box;
    ctx.save();
    ctx.strokeStyle = "rgba(255,255,255,0.07)";
    ctx.lineWidth = 1;
    ctx.fillStyle = "rgba(255,255,255,0.35)";
    ctx.font = "10px " + cssVar("--mono");
    for (let x = Math.ceil(x0 / step) * step; x <= x1 + 1e-9; x += step) {
      const sx = Math.round(v.sx(x)) + 0.5;
      ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, h); ctx.stroke();
      if (labels && x > x0 + 1e-9 && x < x1 - 1e-9) ctx.fillText(+x.toFixed(2), sx + 2, h - 3);
    }
    for (let y = Math.ceil(y0 / step) * step; y <= y1 + 1e-9; y += step) {
      const sy = Math.round(v.sy(y)) + 0.5;
      ctx.beginPath(); ctx.moveTo(0, sy); ctx.lineTo(w, sy); ctx.stroke();
      if (labels && y > y0 + 1e-9 && y < y1 - 1e-9) ctx.fillText(+y.toFixed(2), 3, sy - 3);
    }
    ctx.restore();
  }

  // A data point: filled circle with a dark outline so it reads on any background.
  function drawPoint(ctx, sx, sy, color, r) {
    ctx.beginPath();
    ctx.arc(sx, sy, r || 5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "rgba(0,0,0,0.6)";
    ctx.stroke();
  }

  // Index of the point within `radius` CSS px of (sx, sy), or -1.
  function hitPoint(points, v, sx, sy, radius) {
    let best = -1, bestD = (radius || 14) ** 2;
    points.forEach((p, i) => {
      const d = (v.sx(p.x) - sx) ** 2 + (v.sy(p.y) - sy) ** 2;
      if (d < bestD) { bestD = d; best = i; }
    });
    return best;
  }

  // Pointer position relative to the element, in CSS px.
  function pointerPos(el, e) {
    const r = el.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  // Call `cb` when the element's width really changes. iOS fires "resize"
  // whenever the toolbar hides on scroll, which would redraw for nothing.
  function onWidthChange(el, cb) {
    let last = el.getBoundingClientRect().width, t = null;
    window.addEventListener("resize", () => {
      clearTimeout(t);
      t = setTimeout(() => {
        const w = el.getBoundingClientRect().width;
        if (Math.abs(w - last) >= 1) { last = w; cb(); }
      }, 150);
    });
  }

  // Wire a segmented control (buttons with data-value). Returns a getter.
  function segmented(container, onChange) {
    const buttons = [...container.querySelectorAll("button")];
    let value = buttons.find((b) => b.getAttribute("aria-pressed") === "true").dataset.value;
    buttons.forEach((b) => b.addEventListener("click", () => {
      buttons.forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
      value = b.dataset.value;
      if (onChange) onChange(value);
    }));
    return () => value;
  }

  // A status line. Messages are English templates; the line re-translates
  // itself when the language changes.
  function statusLine(el) {
    let last = null;
    function set(msg, cls, vars) {
      last = [msg, cls, vars];
      el.textContent = msg ? t(msg, vars) : "";
      el.className = "status" + (cls ? " " + cls : "");
    }
    if (window.I18N) window.I18N.onChange(() => last && set.apply(null, last));
    return set;
  }

  // Run `fn` whenever the language is switched (no-op without i18n).
  const onLang = (fn) => window.I18N && window.I18N.onChange(fn);

  window.ML = {
    t, fitCanvas, colormap, fmt, drawLineChart, cssVar, CLASS_COLORS, CLASS_RGB,
    rng, randn, plot2d, drawGrid, drawPoint, hitPoint, pointerPos, onWidthChange, segmented,
    statusLine, onLang,
  };
})();
