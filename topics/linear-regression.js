// Linear regression playground. Formulas match python/linear_regression.py.

(function () {
  "use strict";

  const { plot2d, drawGrid, drawPoint, drawLineChart, hitPoint, pointerPos, onWidthChange, rng, randn, fmt, cssVar } = window.ML;

  // The plot shows x, y in [0, 10]. Gradient descent runs on centred, scaled
  // coordinates u = (x - 5) / 5, v = (y - 5) / 5, where the loss is a round
  // bowl instead of a long valley (see the gradient descent page for why).
  const BOX = [0, 10, 0, 10];
  const toU = (x) => (x - 5) / 5;
  const toV = (y) => (y - 5) / 5;

  const MAX_STEPS = 3000;
  const STEPS_PER_TICK = 2;

  const $ = (id) => document.getElementById(id);
  const canvas = $("plot");
  const chart = $("loss-chart");
  const lossSelect = $("loss");
  const lrInput = $("lr");
  const runBtn = $("run");
  const exactBtn = $("exact");
  const showErrors = $("show-errors");
  const statusEl = $("status");

  let points = [];
  let w = 0, b = 0;          // model in scaled coordinates: v = w * u + b
  let lr = 0.1;
  let steps = 0;
  let losses = [];
  let timer = null;
  let seed = 7;
  let view = null;

  // Prediction in plot units.
  const lineY = (x) => 5 + 5 * (w * toU(x) + b);

  function loss() {
    if (points.length === 0) return NaN;
    let s = 0;
    for (const p of points) {
      const r = lineY(p.x) - p.y;
      s += lossSelect.value === "mse" ? r * r : Math.abs(r);
    }
    return s / points.length;
  }

  // --- The algorithm: one gradient step on MSE or MAE ---------------------

  function step() {
    const n = points.length;
    let gw = 0, gb = 0;
    for (const p of points) {
      const u = toU(p.x);
      const r = w * u + b - toV(p.y);                    // residual of this point
      const pull = lossSelect.value === "mse" ? 2 * r    // MSE: pull grows with distance
                                              : Math.sign(r); // MAE: same pull for everyone
      gw += pull * u;
      gb += pull;
    }
    const rate = lossSelect.value === "mse" ? lr : lr / Math.sqrt(1 + steps / 100);
    w -= rate * gw / n;
    b -= rate * gb / n;
    steps++;
    losses.push(loss());
  }

  // Exact least squares: slope = cov(x, y) / var(x), through the mean point.
  function solveExact() {
    const n = points.length;
    const mx = points.reduce((s, p) => s + p.x, 0) / n;
    const my = points.reduce((s, p) => s + p.y, 0) / n;
    let sxy = 0, sxx = 0;
    for (const p of points) { sxy += (p.x - mx) * (p.y - my); sxx += (p.x - mx) ** 2; }
    if (sxx < 1e-12) return false;
    const slope = sxy / sxx, intercept = my - slope * mx;
    w = slope;
    b = (intercept - 5 + 5 * slope) / 5;
    return true;
  }

  // --- Drawing ----------------------------------------------------------

  function render() {
    const { ctx, w: W, h: H } = view;
    ctx.clearRect(0, 0, W, H);
    drawGrid(view, 2, true);
    const ball = cssVar("--ball");

    if (showErrors.checked) {
      for (const p of points) {
        const sx = view.sx(p.x), sy = view.sy(p.y), ly = view.sy(lineY(p.x));
        if (lossSelect.value === "mse") {
          // The squared error, drawn as an actual square.
          const side = Math.abs(ly - sy);
          const left = sx + side > W ? sx - side : sx;
          ctx.fillStyle = "rgba(255,122,89,0.13)";
          ctx.strokeStyle = "rgba(255,122,89,0.55)";
          ctx.lineWidth = 1;
          ctx.fillRect(left, Math.min(sy, ly), side, side);
          ctx.strokeRect(left, Math.min(sy, ly), side, side);
        } else {
          ctx.strokeStyle = ball;
          ctx.lineWidth = 2;
          ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(sx, ly); ctx.stroke();
        }
      }
    }

    ctx.strokeStyle = cssVar("--accent");
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(view.sx(0), view.sy(lineY(0)));
    ctx.lineTo(view.sx(10), view.sy(lineY(10)));
    ctx.stroke();

    for (const p of points) drawPoint(ctx, view.sx(p.x), view.sy(p.y), "#e7e9ee", 5);

    drawLineChart(chart, losses, { label: lossSelect.value.toUpperCase(), color: ball, empty: "add some points" });
    $("stat-slope").textContent = fmt(w);
    $("stat-intercept").textContent = fmt(lineY(0));
    $("stat-loss-k").textContent = lossSelect.value.toUpperCase();
    $("stat-loss").textContent = points.length ? fmt(loss()) : "–";
    $("stat-step").textContent = steps;
    exactBtn.disabled = lossSelect.value !== "mse";
  }

  function setStatus(msg, cls) {
    statusEl.textContent = msg || "";
    statusEl.className = "status" + (cls ? " " + cls : "");
  }

  // --- Controls ---------------------------------------------------------

  function restartHistory() {
    steps = 0;
    losses = points.length ? [loss()] : [];
  }

  function tick() {
    if (points.length < 2) { stop(); setStatus("Add at least two points."); return; }
    // MAE needs many small steps, so it gets more of them per frame.
    const n = lossSelect.value === "mse" ? STEPS_PER_TICK : 4 * STEPS_PER_TICK;
    for (let i = 0; i < n; i++) {
      const before = losses[losses.length - 1];
      step();
      const after = losses[losses.length - 1];
      if (!isFinite(after) || Math.abs(w) > 1e6) {
        stop(); setStatus("Diverged: the learning rate is too large.", "bad"); break;
      }
      if (lossSelect.value === "mse" && Math.abs(before - after) < 1e-10 * Math.max(1, after)) {
        stop(); setStatus("Converged after " + steps + " steps.", "good"); break;
      }
      if (steps >= MAX_STEPS) {
        stop(); setStatus("Stopped after " + MAX_STEPS + " steps.", ""); break;
      }
    }
    render();
  }

  function run() {
    if (timer) { stop(); return; }
    setStatus("");
    runBtn.textContent = "Pause";
    timer = setInterval(tick, 30);
  }

  function stop() {
    clearInterval(timer);
    timer = null;
    runBtn.textContent = "Run";
  }

  function resetLine() {
    stop();
    w = 0; b = 0;
    restartHistory();
    setStatus("");
    render();
  }

  function randomData() {
    const rand = rng(seed++);
    points = [];
    for (let i = 0; i < 14; i++) {
      const x = 0.5 + rand() * 9;
      const y = 0.55 * x + 2 + 0.7 * randn(rand);
      points.push({ x, y: Math.min(9.8, Math.max(0.2, y)) });
    }
    resetLine();
  }

  function setLr(value) {
    lr = value;
    lrInput.value = Math.log10(value);
    $("lr-out").textContent = value < 0.01 ? value.toFixed(4) : value.toFixed(3);
  }

  lrInput.addEventListener("input", () => setLr(Math.pow(10, parseFloat(lrInput.value))));
  lossSelect.addEventListener("change", () => { restartHistory(); setStatus(""); render(); });
  showErrors.addEventListener("change", render);
  runBtn.addEventListener("click", run);
  exactBtn.addEventListener("click", () => {
    stop();
    if (!solveExact()) { setStatus("Need points with at least two different x values."); return; }
    losses.push(loss());
    setStatus("Exact least-squares solution, no iterations needed.", "good");
    render();
  });
  $("reset").addEventListener("click", resetLine);
  $("random").addEventListener("click", randomData);
  $("outlier").addEventListener("click", () => {
    points.push({ x: 8 + Math.random(), y: 0.5 + Math.random() * 0.5 });
    restartHistory();
    setStatus("Outlier added. Run again and compare MSE with MAE.");
    render();
  });
  $("clear").addEventListener("click", () => { points = []; resetLine(); });

  // Tap empty space to add a point, tap a point to remove it.
  canvas.addEventListener("pointerdown", (e) => {
    const [sx, sy] = pointerPos(canvas, e);
    const i = hitPoint(points, view, sx, sy);
    if (i >= 0) points.splice(i, 1);
    else points.push({ x: view.wx(sx), y: view.wy(sy) });
    restartHistory();
    render();
  });

  function resize() { view = plot2d(canvas, BOX); render(); }
  onWidthChange(canvas, resize);

  setLr(lr);
  view = plot2d(canvas, BOX);
  randomData();
})();
