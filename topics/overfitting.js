// Overfitting playground: polynomial regression of growing degree.
// Formulas match python/overfitting.py.

(function () {
  "use strict";

  const { plot2d, drawGrid, drawPoint, fitCanvas, onWidthChange, rng, randn, fmt, cssVar, onLang, t } = window.ML;

  const BOX = [0, 1, -2, 2];
  const NOISE = 0.3;
  const MAX_DEGREE = 15;
  const N_TEST = 200;
  const N_CLOUD = 20;

  const $ = (id) => document.getElementById(id);
  const canvas = $("plot");
  const errCanvas = $("error-chart");
  const degreeInput = $("degree");
  const nInput = $("n-train");
  const l2Input = $("l2");
  const showTest = $("show-test");
  const showCloud = $("show-cloud");

  const truth = (x) => Math.sin(2 * Math.PI * x);

  let sampleSeed = 1;
  let train = null;
  const test = sample(N_TEST, 999);   // fixed: new data the model never trains on
  let view = null;

  function sample(n, seed) {
    const rand = rng(seed);
    const xs = [], ys = [];
    for (let i = 0; i < n; i++) {
      const x = rand();
      xs.push(x);
      ys.push(truth(x) + NOISE * randn(rand));
    }
    return { xs, ys };
  }

  // --- The model: polynomial = weighted sum of Legendre polynomials ----------

  function features(x, degree) {
    const t = 2 * x - 1;
    const p = [1, t];
    for (let k = 1; k < degree; k++) p.push(((2 * k + 1) * t * p[k] - k * p[k - 1]) / (k + 1));
    return p.slice(0, degree + 1);
  }

  // Solve A w = b by Gaussian elimination with partial pivoting.
  function solve(A, b) {
    const n = b.length;
    const M = A.map((row, i) => [...row, b[i]]);
    for (let col = 0; col < n; col++) {
      let piv = col;
      for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
      [M[col], M[piv]] = [M[piv], M[col]];
      for (let r = col + 1; r < n; r++) {
        const f = M[r][col] / M[col][col];
        for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
      }
    }
    const w = new Array(n).fill(0);
    for (let r = n - 1; r >= 0; r--) {
      let s = M[r][n];
      for (let c = r + 1; c < n; c++) s -= M[r][c] * w[c];
      w[r] = s / M[r][r];
    }
    return w;
  }

  // Ridge regression: minimize mean squared error + l2 * |w|^2, solved exactly.
  function fit(data, degree, l2) {
    const m = degree + 1, n = data.xs.length;
    const A = Array.from({ length: m }, () => new Array(m).fill(0));
    const b = new Array(m).fill(0);
    data.xs.forEach((x, i) => {
      const f = features(x, degree);
      for (let r = 0; r < m; r++) {
        b[r] += (f[r] * data.ys[i]) / n;
        for (let c = 0; c < m; c++) A[r][c] += (f[r] * f[c]) / n;
      }
    });
    for (let r = 0; r < m; r++) A[r][r] += l2 + 1e-10;   // tiny stabilizer, as in the Python version
    return solve(A, b);
  }

  function predict(w, x) {
    const f = features(x, w.length - 1);
    let s = 0;
    for (let i = 0; i < w.length; i++) s += w[i] * f[i];
    return s;
  }

  function mse(w, data) {
    let s = 0;
    data.xs.forEach((x, i) => { s += (predict(w, x) - data.ys[i]) ** 2; });
    return s / data.xs.length;
  }

  // --- Drawing ----------------------------------------------------------

  function curve(fn, color, width, dash) {
    const { ctx, w: W, h: H } = view;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath();
    for (let i = 0; i <= 200; i++) {
      const x = i / 200;
      const sy = Math.max(-H, Math.min(2 * H, view.sy(fn(x))));   // keep wild curves drawable
      if (i === 0) ctx.moveTo(view.sx(x), sy); else ctx.lineTo(view.sx(x), sy);
    }
    ctx.stroke();
    ctx.restore();
  }

  function l2Value() {
    const v = parseFloat(l2Input.value);
    return v <= -7 ? 0 : Math.pow(10, v);
  }

  function drawErrorChart(trainErr, testErr, degree) {
    const { ctx, w: W, h: H } = fitCanvas(errCanvas);
    const pad = { l: 34, r: 8, t: 10, b: 20 };
    const lo = -3, hi = 2;                                   // log10 of MSE shown: 0.001 .. 100
    const X = (d) => pad.l + (d / MAX_DEGREE) * (W - pad.l - pad.r);
    const Y = (v) => pad.t + (1 - (Math.max(lo, Math.min(hi, Math.log10(v))) - lo) / (hi - lo)) * (H - pad.t - pad.b);
    ctx.clearRect(0, 0, W, H);
    ctx.font = "10px " + cssVar("--mono");
    ctx.fillStyle = cssVar("--muted");
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.lineWidth = 1;
    for (let e = lo; e <= hi; e++) {
      const y = Math.round(Y(10 ** e)) + 0.5;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
      ctx.fillText(e >= 0 ? String(10 ** e) : (10 ** e).toFixed(-e), 2, y + 3);
    }
    for (let d = 0; d <= MAX_DEGREE; d += 3) ctx.fillText(String(d), X(d) - 3, H - 5);

    // Noise floor: even the true function has this test error.
    ctx.setLineDash([3, 4]);
    ctx.strokeStyle = "rgba(255,255,255,0.3)";
    ctx.beginPath(); ctx.moveTo(pad.l, Y(NOISE * NOISE)); ctx.lineTo(W - pad.r, Y(NOISE * NOISE)); ctx.stroke();
    ctx.setLineDash([]);

    ctx.strokeStyle = "rgba(255,255,255,0.5)";
    ctx.beginPath(); ctx.moveTo(X(degree), pad.t); ctx.lineTo(X(degree), H - pad.b); ctx.stroke();

    [[trainErr, cssVar("--accent")], [testErr, cssVar("--ball")]].forEach(([vals, color]) => {
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      vals.forEach((v, d) => (d ? ctx.lineTo(X(d), Y(v)) : ctx.moveTo(X(d), Y(v))));
      ctx.stroke();
      ctx.beginPath(); ctx.arc(X(degree), Y(vals[degree]), 4, 0, Math.PI * 2); ctx.fill();
    });
  }

  function render() {
    const { ctx, w: W, h: H } = view;
    const degree = parseInt(degreeInput.value, 10);
    const l2 = l2Value();
    $("degree-out").textContent = degree;
    $("n-out").textContent = nInput.value;
    $("l2-out").textContent = l2 === 0 ? t("off") : l2.toExponential(0);

    ctx.clearRect(0, 0, W, H);
    drawGrid(view, 0.5, false);

    if (showTest.checked) {
      test.xs.forEach((x, i) => {
        ctx.fillStyle = "rgba(255,122,89,0.35)";
        ctx.beginPath(); ctx.arc(view.sx(x), view.sy(test.ys[i]), 2.2, 0, Math.PI * 2); ctx.fill();
      });
    }

    // Many training sets -> many fitted curves: their spread is the variance,
    // the gap between their average and the truth is the bias.
    if (showCloud.checked) {
      const fits = [];
      for (let i = 0; i < N_CLOUD; i++) fits.push(fit(sample(train.xs.length, 5000 + sampleSeed * 100 + i), degree, l2));
      fits.forEach((w) => curve((x) => predict(w, x), "rgba(255,255,255,0.16)", 1));
      curve((x) => fits.reduce((s, w) => s + predict(w, x), 0) / fits.length, cssVar("--ball"), 2.5, [7, 5]);
    }

    curve(truth, "rgba(255,255,255,0.55)", 1.5, [4, 4]);
    const w = fit(train, degree, l2);
    curve((x) => predict(w, x), cssVar("--accent"), 3);
    train.xs.forEach((x, i) => drawPoint(ctx, view.sx(x), view.sy(train.ys[i]), "#e7e9ee", 4.5));

    const trainErr = [], testErr = [];
    for (let d = 0; d <= MAX_DEGREE; d++) {
      const wd = d === degree ? w : fit(train, d, l2);
      trainErr.push(mse(wd, train));
      testErr.push(mse(wd, test));
    }
    drawErrorChart(trainErr, testErr, degree);

    $("stat-train").textContent = fmt(trainErr[degree]);
    $("stat-test").textContent = fmt(testErr[degree]);
    $("stat-params").textContent = degree + 1 + " / " + train.xs.length;
    $("stat-w").textContent = fmt(Math.hypot(...w));
  }

  function newSample() {
    sampleSeed++;
    train = sample(parseInt(nInput.value, 10), sampleSeed);
    render();
  }

  degreeInput.addEventListener("input", render);
  l2Input.addEventListener("input", render);
  nInput.addEventListener("input", () => { train = sample(parseInt(nInput.value, 10), sampleSeed); render(); });
  showTest.addEventListener("change", render);
  showCloud.addEventListener("change", render);
  $("resample").addEventListener("click", newSample);

  onWidthChange(canvas, () => { view = plot2d(canvas, BOX); render(); });

  onLang(render);

  view = plot2d(canvas, BOX);
  train = sample(parseInt(nInput.value, 10), sampleSeed);
  render();
})();
