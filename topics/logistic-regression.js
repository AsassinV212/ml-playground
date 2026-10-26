// Logistic regression playground. Formulas match python/logistic_regression.py.

(function () {
  "use strict";

  const {
    plot2d, drawGrid, drawPoint, drawLineChart, hitPoint, pointerPos, onWidthChange,
    segmented, rng, randn, fmt, cssVar, CLASS_COLORS, CLASS_RGB, statusLine, onLang, t,
  } = window.ML;

  const BOX = [-5, 5, -5, 5];
  const LR = 0.3;
  const MAX_STEPS = 4000;
  const STEPS_PER_TICK = 4;
  const Z90 = Math.log(9);   // sigmoid(Z90) = 0.9

  const $ = (id) => document.getElementById(id);
  const canvas = $("plot");
  const chart = $("loss-chart");
  const l2Input = $("l2");
  const runBtn = $("run");
  const setStatus = statusLine($("status"));
  const currentClass = segmented($("class-picker"));

  let points = [];           // { x, y, c } with c = 0 (coral) or 1 (blue)
  let w1 = 0, w2 = 0, b = 0;
  let l2 = 0;
  let steps = 0;
  let losses = [];
  let timer = null;
  let view = null;

  // Numerically safe sigmoid.
  function sigmoid(z) {
    if (z >= 0) return 1 / (1 + Math.exp(-z));
    const e = Math.exp(z);
    return e / (1 + e);
  }
  const prob = (x, y) => sigmoid(w1 * x + w2 * y + b);   // P(blue | x, y)

  function logLoss() {
    let s = 0;
    for (const p of points) {
      const q = Math.min(1 - 1e-12, Math.max(1e-12, prob(p.x, p.y)));
      s += p.c === 1 ? -Math.log(q) : -Math.log(1 - q);
    }
    return s / points.length + 0.5 * l2 * (w1 * w1 + w2 * w2);
  }

  function accuracy() {
    let ok = 0;
    for (const p of points) ok += (prob(p.x, p.y) >= 0.5 ? 1 : 0) === p.c ? 1 : 0;
    return ok / points.length;
  }

  // --- The algorithm: one gradient step on the log loss -------------------

  function step() {
    let g1 = 0, g2 = 0, gb = 0;
    for (const p of points) {
      const err = prob(p.x, p.y) - p.c;    // predicted probability minus the true label
      g1 += err * p.x;
      g2 += err * p.y;
      gb += err;
    }
    const n = points.length;
    g1 = g1 / n + l2 * w1;
    g2 = g2 / n + l2 * w2;
    gb = gb / n;
    w1 -= LR * g1;
    w2 -= LR * g2;
    b -= LR * gb;
    steps++;
    losses.push(logLoss());
    return Math.hypot(g1, g2, gb);
  }

  // --- Drawing ----------------------------------------------------------

  // Probability map: colored by the likelier class, brighter where the model is surer.
  let small = document.createElement("canvas");
  function drawProbability() {
    const { ctx, w: W, h: H } = view;
    const cw = Math.ceil(W / 4), ch = Math.ceil(H / 4);
    small.width = cw; small.height = ch;
    const sctx = small.getContext("2d");
    const img = sctx.createImageData(cw, ch);
    const bg = [32, 36, 46];
    for (let j = 0; j < ch; j++) {
      const y = view.wy((j + 0.5) * 4);
      for (let i = 0; i < cw; i++) {
        const q = prob(view.wx((i + 0.5) * 4), y);
        const c = CLASS_RGB[q >= 0.5 ? 1 : 0];
        const s = 0.5 * Math.abs(2 * q - 1);
        const k = (j * cw + i) * 4;
        img.data[k] = bg[0] + (c[0] - bg[0]) * s;
        img.data[k + 1] = bg[1] + (c[1] - bg[1]) * s;
        img.data[k + 2] = bg[2] + (c[2] - bg[2]) * s;
        img.data[k + 3] = 255;
      }
    }
    sctx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(small, 0, 0, W, H);
  }

  // The straight line where w1 x + w2 y + b = z.
  function drawLevel(z, dashed) {
    const nn = w1 * w1 + w2 * w2;
    if (nn < 1e-12) return;
    const { ctx } = view;
    const px = ((z - b) * w1) / nn, py = ((z - b) * w2) / nn;   // closest point to the origin
    const dx = -w2 / Math.sqrt(nn), dy = w1 / Math.sqrt(nn);    // direction along the line
    ctx.save();
    ctx.strokeStyle = dashed ? "rgba(255,255,255,0.45)" : "#fff";
    ctx.lineWidth = dashed ? 1.5 : 2.5;
    if (dashed) ctx.setLineDash([6, 6]);
    ctx.beginPath();
    ctx.moveTo(view.sx(px - 30 * dx), view.sy(py - 30 * dy));
    ctx.lineTo(view.sx(px + 30 * dx), view.sy(py + 30 * dy));
    ctx.stroke();
    ctx.restore();
  }

  function render() {
    const { ctx, w: W, h: H } = view;
    ctx.clearRect(0, 0, W, H);
    drawProbability();
    drawGrid(view, 1, false);
    drawLevel(-Z90, true);
    drawLevel(Z90, true);
    drawLevel(0, false);
    for (const p of points) drawPoint(ctx, view.sx(p.x), view.sy(p.y), CLASS_COLORS[p.c], 5.5);

    drawLineChart(chart, losses, { label: "log loss", color: cssVar("--ball"), empty: "add points of both colors" });
    const has = points.length > 0;
    $("stat-loss").textContent = has ? fmt(logLoss()) : "–";
    $("stat-acc").textContent = has ? Math.round(accuracy() * 100) + "%" : "–";
    $("stat-w").textContent = fmt(Math.hypot(w1, w2));
    $("stat-step").textContent = steps;
  }

  // --- Controls ---------------------------------------------------------

  function bothClasses() {
    return points.some((p) => p.c === 0) && points.some((p) => p.c === 1);
  }

  function restartHistory() {
    steps = 0;
    losses = points.length ? [logLoss()] : [];
  }

  function tick() {
    if (!bothClasses()) { stop(); setStatus("Add points of both colors."); render(); return; }
    for (let i = 0; i < STEPS_PER_TICK; i++) {
      const g = step();
      if (g < 1e-5) { stop(); setStatus("Converged after {n} steps.", "good", { n: steps }); break; }
      if (steps >= MAX_STEPS) {
        stop();
        if (l2 === 0 && accuracy() === 1) {
          setStatus("Still not converged: the classes are separable, so the weights keep growing and the edge keeps getting sharper. Turn on L2.", "bad");
        } else {
          setStatus("Stopped after {n} steps.", "", { n: MAX_STEPS });
        }
        break;
      }
    }
    render();
  }

  function run() {
    if (timer) { stop(); return; }
    setStatus("");
    runBtn.textContent = t("Pause");
    timer = setInterval(tick, 30);
  }

  function stop() {
    clearInterval(timer);
    timer = null;
    runBtn.textContent = t("Run");
  }

  function resetModel() {
    stop();
    w1 = 0; w2 = 0; b = 0;
    restartHistory();
    setStatus("");
    render();
  }

  // Two Gaussian clouds; `spread` controls how much they overlap.
  function makeData(spread, seed) {
    const rand = rng(seed);
    points = [];
    for (let c = 0; c < 2; c++) {
      const cx = c ? 1.6 : -1.6, cy = c ? 1.0 : -1.0;
      for (let i = 0; i < 22; i++) {
        points.push({
          x: Math.max(-4.8, Math.min(4.8, cx + spread * randn(rand))),
          y: Math.max(-4.8, Math.min(4.8, cy + spread * randn(rand))),
          c,
        });
      }
    }
    resetModel();
  }

  function setL2(slider) {
    l2 = slider <= -4 ? 0 : Math.pow(10, slider);
    $("l2-out").textContent = l2 === 0 ? t("off") : l2 < 0.01 ? l2.toFixed(4) : l2.toFixed(3);
    if (points.length) { restartHistory(); render(); }
  }

  l2Input.addEventListener("input", () => setL2(parseFloat(l2Input.value)));
  runBtn.addEventListener("click", run);
  $("step").addEventListener("click", () => {
    stop();
    if (!bothClasses()) { setStatus("Add points of both colors."); return; }
    step(); setStatus(""); render();
  });
  $("reset").addEventListener("click", resetModel);
  $("separable").addEventListener("click", () => makeData(0.7, 3));
  $("overlap").addEventListener("click", () => makeData(1.5, 5));
  $("clear").addEventListener("click", () => { points = []; resetModel(); });

  canvas.addEventListener("pointerdown", (e) => {
    const [sx, sy] = pointerPos(canvas, e);
    const i = hitPoint(points, view, sx, sy);
    if (i >= 0) points.splice(i, 1);
    else points.push({ x: view.wx(sx), y: view.wy(sy), c: +currentClass() });
    restartHistory();
    render();
  });

  onWidthChange(canvas, () => { view = plot2d(canvas, BOX); render(); });

  onLang(() => {
    runBtn.textContent = t(timer ? "Pause" : "Run");
    if (l2 === 0) $("l2-out").textContent = t("off");
    render();
  });

  view = plot2d(canvas, BOX);
  setL2(parseFloat(l2Input.value));
  makeData(0.7, 3);
})();
