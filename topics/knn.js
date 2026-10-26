// k-nearest neighbors playground. Logic matches python/knn.py.

(function () {
  "use strict";

  const {
    plot2d, drawGrid, drawPoint, hitPoint, pointerPos, onWidthChange,
    segmented, rng, randn, CLASS_COLORS, CLASS_RGB, statusLine, onLang, t,
  } = window.ML;

  const BOX = [-5, 5, -5, 5];
  const CELL = 3;                         // region map resolution, CSS px
  const NAMES = ["coral", "blue", "yellow"];

  const $ = (id) => document.getElementById(id);
  const canvas = $("plot");
  const kInput = $("k");
  const metricSelect = $("metric");
  const setStatus = statusLine($("status"));

  let points = [];                        // { x, y, c }
  let probe = null;                       // query point in probe mode
  let view = null;
  const mode = segmented($("mode"), (m) => {
    canvas.style.touchAction = m === "probe" ? "none" : "manipulation";
    if (m !== "probe") { probe = null; setStatus(""); render(); }
    else setStatus("Tap or drag on the map to place the query point.");
  });

  const k = () => parseInt(kInput.value, 10);

  // --- The algorithm ------------------------------------------------------

  function dist(ax, ay, bx, by) {
    const dx = ax - bx, dy = ay - by;
    return metricSelect.value === "manhattan" ? Math.abs(dx) + Math.abs(dy) : Math.sqrt(dx * dx + dy * dy);
  }

  // Indices of the k points closest to (x, y), nearest first. `skip` leaves one point out.
  function nearest(x, y, kk, skip) {
    const idx = [], ds = [];
    for (let i = 0; i < points.length; i++) {
      if (i === skip) continue;
      const d = dist(x, y, points[i].x, points[i].y);
      if (idx.length === kk && d >= ds[kk - 1]) continue;
      let j = Math.min(idx.length, kk - 1);              // insertion sort into the top-k list
      while (j > 0 && ds[j - 1] > d) { ds[j] = ds[j - 1]; idx[j] = idx[j - 1]; j--; }
      ds[j] = d; idx[j] = i;
      if (idx.length > kk) { idx.length = kk; ds.length = kk; }
    }
    return { idx, ds };
  }

  // Majority vote; a tie goes to the class whose member is closest.
  function vote(idx) {
    const counts = [0, 0, 0];
    for (const i of idx) counts[points[i].c]++;
    const top = Math.max(...counts);
    for (const i of idx) if (counts[points[i].c] === top) return { c: points[i].c, counts, frac: top / idx.length };
    return null;
  }

  function classify(x, y, skip) {
    const n = points.length - (skip >= 0 ? 1 : 0);
    if (n <= 0) return null;
    return vote(nearest(x, y, Math.min(k(), n), skip).idx);
  }

  // Accuracy on the training points themselves, and leaving each point out in turn.
  function accuracies() {
    let train = 0, loo = 0;
    points.forEach((p, i) => {
      if (classify(p.x, p.y, -1).c === p.c) train++;
      const r = classify(p.x, p.y, i);
      if (r && r.c === p.c) loo++;
    });
    return [train / points.length, loo / points.length];
  }

  // --- Drawing ----------------------------------------------------------

  // The region map is cached: moving only the probe point reuses it.
  const small = document.createElement("canvas");
  function drawRegions(recompute) {
    const { ctx, w: W, h: H } = view;
    ctx.imageSmoothingEnabled = true;
    if (!recompute) { ctx.drawImage(small, 0, 0, W, H); return; }
    const cw = Math.ceil(W / CELL), ch = Math.ceil(H / CELL);
    small.width = cw; small.height = ch;
    const sctx = small.getContext("2d");
    const img = sctx.createImageData(cw, ch);
    const bg = [32, 36, 46];
    for (let j = 0; j < ch; j++) {
      const y = view.wy((j + 0.5) * CELL);
      for (let i = 0; i < cw; i++) {
        const r = classify(view.wx((i + 0.5) * CELL), y, -1);
        const o = (j * cw + i) * 4;
        const c = r ? CLASS_RGB[r.c] : bg;
        const s = r ? 0.18 + 0.32 * r.frac : 0;           // brighter where the vote is unanimous
        img.data[o] = bg[0] + (c[0] - bg[0]) * s;
        img.data[o + 1] = bg[1] + (c[1] - bg[1]) * s;
        img.data[o + 2] = bg[2] + (c[2] - bg[2]) * s;
        img.data[o + 3] = 255;
      }
    }
    sctx.putImageData(img, 0, 0);
    ctx.drawImage(small, 0, 0, W, H);
  }

  function drawProbe() {
    const { ctx } = view;
    const kk = Math.min(k(), points.length);
    const { idx, ds } = nearest(probe.x, probe.y, kk, -1);
    const px = view.sx(probe.x), py = view.sy(probe.y);

    // The neighbourhood: a circle for Euclid, a diamond for Manhattan.
    const r = (ds[kk - 1] || 0) * (view.w / (BOX[1] - BOX[0]));
    ctx.save();
    ctx.strokeStyle = "rgba(255,255,255,0.6)";
    ctx.setLineDash([5, 5]);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    if (metricSelect.value === "manhattan") {
      ctx.moveTo(px, py - r); ctx.lineTo(px + r, py); ctx.lineTo(px, py + r); ctx.lineTo(px - r, py); ctx.closePath();
    } else {
      ctx.arc(px, py, r, 0, Math.PI * 2);
    }
    ctx.stroke();
    ctx.restore();

    ctx.strokeStyle = "rgba(255,255,255,0.8)";
    ctx.lineWidth = 1;
    for (const i of idx) {
      ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(view.sx(points[i].x), view.sy(points[i].y)); ctx.stroke();
    }
    return { idx, result: vote(idx) };
  }

  function render(probeOnly) {
    const { ctx, w: W, h: H } = view;
    ctx.clearRect(0, 0, W, H);
    if (points.length) drawRegions(!probeOnly);
    drawGrid(view, 1, false);

    let neighbours = [];
    if (probe && points.length) {
      const { idx, result } = drawProbe();
      neighbours = idx;
      const parts = result.counts.map((n, c) => (n ? t(NAMES[c]) + " " + n : null)).filter(Boolean);
      setStatus("Neighbours: {votes} → {winner}", "", { votes: parts.join(" · "), winner: t(NAMES[result.c]) });
    }
    for (const p of points) drawPoint(ctx, view.sx(p.x), view.sy(p.y), CLASS_COLORS[p.c], 5);
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    for (const i of neighbours) {
      ctx.beginPath(); ctx.arc(view.sx(points[i].x), view.sy(points[i].y), 8.5, 0, Math.PI * 2); ctx.stroke();
    }
    if (probe) {
      ctx.fillStyle = "#fff";
      ctx.beginPath(); ctx.arc(view.sx(probe.x), view.sy(probe.y), 6, 0, Math.PI * 2); ctx.fill();
    }

    if (probeOnly) return;
    $("k-out").textContent = k();
    $("stat-n").textContent = points.length;
    if (points.length) {
      const [train, loo] = accuracies();
      $("stat-train").textContent = Math.round(train * 100) + "%";
      $("stat-loo").textContent = Math.round(loo * 100) + "%";
    } else {
      $("stat-train").textContent = $("stat-loo").textContent = "–";
    }
    $("stat-k").textContent = k();
  }

  // --- Data -------------------------------------------------------------

  function blobs() {
    const rand = rng(11);
    const centres = [[-2, -1.5], [2, -1.5], [0, 2.2]];
    points = [];
    centres.forEach(([cx, cy], c) => {
      for (let i = 0; i < 20; i++) points.push({ x: cx + 1.1 * randn(rand), y: cy + 1.1 * randn(rand), c });
    });
    probe = null;
    render();
  }

  // Two classes split by a wavy border, with 12% of labels flipped at random.
  function noisy() {
    const rand = rng(21);
    points = [];
    for (let i = 0; i < 110; i++) {
      const x = -4.7 + rand() * 9.4, y = -4.7 + rand() * 9.4;
      let c = y > 1.6 * Math.sin(x * 0.9) ? 1 : 0;
      if (rand() < 0.12) c = 1 - c;
      points.push({ x, y, c });
    }
    probe = null;
    render();
  }

  // --- Controls ---------------------------------------------------------

  kInput.addEventListener("input", () => render());
  metricSelect.addEventListener("change", () => render());
  $("blobs").addEventListener("click", blobs);
  $("noisy").addEventListener("click", noisy);
  $("clear").addEventListener("click", () => { points = []; probe = null; setStatus(""); render(); });

  function placeProbe(e) {
    const [sx, sy] = pointerPos(canvas, e);
    probe = { x: view.wx(sx), y: view.wy(sy) };
    render(true);
  }

  canvas.addEventListener("pointerdown", (e) => {
    if (mode() === "probe") { canvas.setPointerCapture(e.pointerId); placeProbe(e); return; }
    const [sx, sy] = pointerPos(canvas, e);
    const i = hitPoint(points, view, sx, sy);
    if (i >= 0) points.splice(i, 1);
    else points.push({ x: view.wx(sx), y: view.wy(sy), c: +mode() });
    render();
  });
  canvas.addEventListener("pointermove", (e) => {
    if (mode() === "probe" && e.buttons) placeProbe(e);
  });

  onWidthChange(canvas, () => { view = plot2d(canvas, BOX); render(); });

  onLang(() => render(true));

  view = plot2d(canvas, BOX);
  blobs();
})();
