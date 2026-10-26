// Gradient descent playground. Surfaces and update rule match python/gradient_descent.py.

(function () {
  "use strict";

  const { fitCanvas, colormap, fmt, drawLineChart } = window.ML;

  // --- Loss surfaces ------------------------------------------------------
  // f(x, y) is the loss, grad(x, y) its gradient [df/dx, df/dy].

  const SURFACES = {
    bowl: {
      f: (x, y) => x * x + y * y,
      grad: (x, y) => [2 * x, 2 * y],
      view: [-3, 3, -3, 3],
      start: [2.4, -1.8],
      lr: 0.1,
      minima: [{ x: 0, y: 0, label: "minimum" }],
    },
    valley: {
      f: (x, y) => 0.5 * x * x + 5 * y * y,
      grad: (x, y) => [x, 10 * y],
      view: [-3, 3, -3, 3],
      start: [-2.6, 1.2],
      lr: 0.15,
      minima: [{ x: 0, y: 0, label: "minimum" }],
    },
    local: {
      f: (x, y) => 0.25 * x ** 4 - x * x + 0.3 * x + 0.5 * y * y,
      grad: (x, y) => [x ** 3 - 2 * x + 0.3, y],
      view: [-2.5, 2.5, -2.5, 2.5],
      start: [2.2, 1.8],
      lr: 0.05,
      minima: [
        { x: -1.4795, y: 0, label: "global min" },
        { x: 1.3268, y: 0, label: "local min" },
      ],
    },
  };

  const MAX_STEPS = 500;
  const STEP_MS = 60;        // animation speed of "run"
  const CONVERGED = 1e-4;    // stop when |gradient| is this small

  // --- DOM ----------------------------------------------------------------

  const $ = (id) => document.getElementById(id);
  const surfaceCanvas = $("surface");
  const chartCanvas = $("loss-chart");
  const lrInput = $("lr");
  const lrOut = $("lr-out");
  const surfaceSelect = $("surface-select");
  const runBtn = $("run");
  const statusEl = $("status");

  // --- State --------------------------------------------------------------

  let surface = SURFACES.bowl;
  let start = surface.start.slice();
  let path = [];             // points visited: [[x, y], ...]
  let losses = [];
  let lr = surface.lr;
  let timer = null;
  let heatmap = null;        // offscreen canvas with the pre-rendered surface
  let view = null;           // { ctx, w, h } of the main canvas

  // --- The algorithm: one step of gradient descent -------------------------

  function step() {
    const [x, y] = path[path.length - 1];
    const [gx, gy] = surface.grad(x, y);            // uphill direction
    const next = [x - lr * gx, y - lr * gy];        // move the other way
    path.push(next);
    losses.push(surface.f(next[0], next[1]));
    return next;
  }

  // Checks after each step: did we arrive, blow up, or run out of steps?
  function checkState() {
    const [x, y] = path[path.length - 1];
    const [gx, gy] = surface.grad(x, y);
    const [x0, x1] = surface.view;
    const far = 20 * (x1 - x0);
    if (!isFinite(x) || !isFinite(y) || Math.abs(x) > far || Math.abs(y) > far) {
      return { done: true, cls: "bad", msg: "Diverged: each step overshoots further. Lower the learning rate." };
    }
    if (Math.hypot(gx, gy) < CONVERGED) {
      return { done: true, cls: "good", msg: "Converged after " + (path.length - 1) + " steps." };
    }
    if (path.length - 1 >= MAX_STEPS) {
      return { done: true, cls: "", msg: "Stopped after " + MAX_STEPS + " steps. Still moving: the learning rate may be too small." };
    }
    return { done: false };
  }

  // --- Coordinates ----------------------------------------------------------

  function toScreen(x, y) {
    const [x0, x1, y0, y1] = surface.view;
    return [((x - x0) / (x1 - x0)) * view.w, (1 - (y - y0) / (y1 - y0)) * view.h];
  }
  function toWorld(sx, sy) {
    const [x0, x1, y0, y1] = surface.view;
    return [x0 + (sx / view.w) * (x1 - x0), y0 + (1 - sy / view.h) * (y1 - y0)];
  }

  // --- Drawing ---------------------------------------------------------------

  // Heatmap + contour lines, computed once per surface and size.
  function buildHeatmap() {
    const W = surfaceCanvas.width, H = surfaceCanvas.height;
    const [x0, x1, y0, y1] = surface.view;
    const vals = new Float64Array(W * H);
    let lo = Infinity, hi = -Infinity;
    for (let j = 0; j < H; j++) {
      const y = y1 - ((j + 0.5) / H) * (y1 - y0);
      for (let i = 0; i < W; i++) {
        const v = surface.f(x0 + ((i + 0.5) / W) * (x1 - x0), y);
        vals[j * W + i] = v;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    // sqrt spreads the colors evenly from the bottom of the bowl to its rim.
    const BANDS = 14;
    const band = new Int16Array(W * H);
    const tArr = new Float32Array(W * H);
    for (let k = 0; k < vals.length; k++) {
      const t = Math.sqrt((vals[k] - lo) / (hi - lo));
      tArr[k] = t;
      band[k] = Math.floor(t * BANDS);
    }
    const off = document.createElement("canvas");
    off.width = W; off.height = H;
    const octx = off.getContext("2d");
    const img = octx.createImageData(W, H);
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const k = j * W + i;
        let [r, g, b] = colormap(tArr[k]);
        // A contour line wherever the band changes between neighbouring pixels.
        const edge = (i + 1 < W && band[k + 1] !== band[k]) || (j + 1 < H && band[k + W] !== band[k]);
        if (edge) { r += (255 - r) * 0.35; g += (255 - g) * 0.35; b += (255 - b) * 0.35; }
        const p = k * 4;
        img.data[p] = r; img.data[p + 1] = g; img.data[p + 2] = b; img.data[p + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);
    heatmap = off;
  }

  function drawSurface() {
    const { ctx, w, h } = view;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(heatmap, 0, 0);
    ctx.restore();

    // Minima markers.
    ctx.font = "12px " + getComputedStyle(document.body).fontFamily;
    ctx.textBaseline = "top";
    for (const m of surface.minima) {
      const [sx, sy] = toScreen(m.x, m.y);
      ctx.strokeStyle = "rgba(255,255,255,0.8)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(sx - 5, sy - 5); ctx.lineTo(sx + 5, sy + 5);
      ctx.moveTo(sx + 5, sy - 5); ctx.lineTo(sx - 5, sy + 5);
      ctx.stroke();
      ctx.fillStyle = "rgba(255,255,255,0.75)";
      ctx.textAlign = sx > w - 60 ? "right" : "center";
      ctx.fillText(m.label, sx, sy + 8);
    }
    ctx.textAlign = "left";

    // Path the ball has taken.
    const ball = getComputedStyle(document.documentElement).getPropertyValue("--ball").trim();
    const pts = path.map((p) => toScreen(p[0], p[1]));
    ctx.strokeStyle = "rgba(255,255,255,0.85)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    pts.forEach(([sx, sy], i) => (i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy)));
    ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,0.9)";
    for (let i = 1; i < pts.length - 1; i++) {
      ctx.beginPath();
      ctx.arc(pts[i][0], pts[i][1], 2.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Start ring and the ball itself.
    const [s0x, s0y] = pts[0];
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(s0x, s0y, 6, 0, Math.PI * 2);
    ctx.stroke();
    const [bx, by] = pts[pts.length - 1];
    ctx.fillStyle = ball;
    ctx.beginPath();
    ctx.arc(bx, by, 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

  function render() {
    drawSurface();
    drawLineChart(chartCanvas, losses, { label: "loss", color: getComputedStyle(document.documentElement).getPropertyValue("--ball").trim() });
    const [x, y] = path[path.length - 1];
    const [gx, gy] = surface.grad(x, y);
    $("stat-iter").textContent = path.length - 1;
    $("stat-loss").textContent = fmt(losses[losses.length - 1]);
    $("stat-grad").textContent = fmt(Math.hypot(gx, gy));
    $("stat-pos").textContent = "(" + x.toFixed(2) + ", " + y.toFixed(2) + ")";
  }

  function setStatus(msg, cls) {
    statusEl.textContent = msg || "";
    statusEl.className = "status" + (cls ? " " + cls : "");
  }

  // --- Controls ---------------------------------------------------------------

  function reset() {
    stop();
    path = [start.slice()];
    losses = [surface.f(start[0], start[1])];
    setStatus("Tap the map to choose a starting point.");
    render();
  }

  function doStep() {
    if (checkState().done && path.length > 1) return;
    step();
    const s = checkState();
    if (s.done) { stop(); setStatus(s.msg, s.cls); } else setStatus("");
    render();
  }

  function run() {
    if (timer) { stop(); return; }
    if (checkState().done && path.length > 1) reset();
    runBtn.textContent = "Pause";
    timer = setInterval(() => {
      doStep();
    }, STEP_MS);
  }

  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
    runBtn.textContent = "Run";
  }

  function setLr(value) {
    lr = value;
    lrInput.value = Math.log10(value);
    lrOut.textContent = value < 0.01 ? value.toFixed(4) : value.toFixed(3);
  }

  function chooseSurface(name) {
    surface = SURFACES[name];
    start = surface.start.slice();
    setLr(surface.lr);
    resize();
  }

  function resize() {
    view = fitCanvas(surfaceCanvas);
    buildHeatmap();
    reset();
  }

  // Slider works in log10 space: equal finger movement = equal *ratio* of learning rates.
  lrInput.addEventListener("input", () => setLr(Math.pow(10, parseFloat(lrInput.value))));
  surfaceSelect.addEventListener("change", () => chooseSurface(surfaceSelect.value));
  $("step").addEventListener("click", () => { stop(); doStep(); });
  runBtn.addEventListener("click", run);
  $("reset").addEventListener("click", reset);

  surfaceCanvas.addEventListener("pointerdown", (e) => {
    const rect = surfaceCanvas.getBoundingClientRect();
    start = toWorld(e.clientX - rect.left, e.clientY - rect.top);
    reset();
    setStatus("New start point. Press Step or Run.");
  });

  let resizeTimer = null;
  let lastWidth = surfaceCanvas.getBoundingClientRect().width;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      // iOS fires resize when the toolbar hides on scroll; only redraw on real width changes.
      const w = surfaceCanvas.getBoundingClientRect().width;
      if (Math.abs(w - lastWidth) < 1) return;
      lastWidth = w;
      view = fitCanvas(surfaceCanvas);
      buildHeatmap();
      render();
    }, 150);
  });

  chooseSurface(surfaceSelect.value);
})();
