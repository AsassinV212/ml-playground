// Decision tree playground. Logic matches python/decision_tree.py.

(function () {
  "use strict";

  const {
    plot2d, drawGrid, drawPoint, hitPoint, pointerPos, onWidthChange,
    segmented, rng, randn, CLASS_COLORS, CLASS_RGB,
  } = window.ML;

  const BOX = [-5, 5, -5, 5];
  const N_CLASSES = 3;
  const NAMES = ["coral", "blue", "yellow"];
  const AXES = ["x", "y"];

  const $ = (id) => document.getElementById(id);
  const canvas = $("plot");
  const depthInput = $("depth");
  const leafInput = $("min-leaf");
  const criterionSelect = $("criterion");
  const currentClass = segmented($("class-picker"));

  let points = [];          // { x, y, c }
  let tree = null;
  let view = null;

  // --- Impurity -----------------------------------------------------------

  function gini(counts, n) {
    let s = 0;
    for (const k of counts) s += (k / n) ** 2;
    return 1 - s;
  }

  function entropy(counts, n) {
    let s = 0;
    for (const k of counts) if (k > 0) s -= (k / n) * Math.log2(k / n);
    return s;
  }

  // --- The algorithm: greedy recursive splitting ----------------------------

  function bestSplit(pts, impurity, minLeaf) {
    const n = pts.length;
    const total = new Array(N_CLASSES).fill(0);
    for (const p of pts) total[p.c]++;
    let best = null, bestScore = impurity(total, n) - 1e-12;   // a split must actually help
    for (let f = 0; f < 2; f++) {
      const key = AXES[f];
      const sorted = pts.map((p, i) => [p, i]).sort((a, b) => a[0][key] - b[0][key] || a[1] - b[1]).map((e) => e[0]);
      const left = new Array(N_CLASSES).fill(0);
      for (let i = 1; i < n; i++) {
        left[sorted[i - 1].c]++;                                  // move one point to the left side
        if (i < minLeaf || n - i < minLeaf || sorted[i - 1][key] === sorted[i][key]) continue;
        const right = total.map((t, c) => t - left[c]);
        const score = (i * impurity(left, i) + (n - i) * impurity(right, n - i)) / n;
        if (score < bestScore) {
          bestScore = score;
          best = { f, t: (sorted[i - 1][key] + sorted[i][key]) / 2 };
        }
      }
    }
    return best;
  }

  function build(pts, depth, opts) {
    const counts = new Array(N_CLASSES).fill(0);
    for (const p of pts) counts[p.c]++;
    const label = counts.indexOf(Math.max(...counts));
    const leaf = { leaf: true, label, counts, n: pts.length };
    if (depth >= opts.maxDepth || counts.filter((k) => k > 0).length <= 1) return leaf;
    const split = bestSplit(pts, opts.impurity, opts.minLeaf);
    if (!split) return leaf;
    const key = AXES[split.f];
    return {
      leaf: false, f: split.f, t: split.t, n: pts.length,
      left: build(pts.filter((p) => p[key] < split.t), depth + 1, opts),
      right: build(pts.filter((p) => p[key] >= split.t), depth + 1, opts),
    };
  }

  function predict(node, p) {
    while (!node.leaf) node = p[AXES[node.f]] < node.t ? node.left : node.right;
    return node.label;
  }

  // --- Drawing ----------------------------------------------------------

  // Walk the tree with the box of plane each node is responsible for.
  function drawNode(node, box, depth) {
    const { ctx } = view;
    const [x0, x1, y0, y1] = box;
    if (node.leaf) {
      if (node.n === 0) return;
      const purity = node.counts[node.label] / node.n;
      const c = CLASS_RGB[node.label];
      ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${0.12 + 0.3 * purity})`;
      ctx.fillRect(view.sx(x0), view.sy(y1), view.sx(x1) - view.sx(x0), view.sy(y0) - view.sy(y1));
      return;
    }
    if (node.f === 0) {
      drawNode(node.left, [x0, node.t, y0, y1], depth + 1);
      drawNode(node.right, [node.t, x1, y0, y1], depth + 1);
    } else {
      drawNode(node.left, [x0, x1, y0, node.t], depth + 1);
      drawNode(node.right, [x0, x1, node.t, y1], depth + 1);
    }
    // The cut itself; earlier (more important) cuts are thicker.
    ctx.strokeStyle = "rgba(255,255,255,0.85)";
    ctx.lineWidth = Math.max(1, 3.5 - depth * 0.6);
    ctx.beginPath();
    if (node.f === 0) { ctx.moveTo(view.sx(node.t), view.sy(y0)); ctx.lineTo(view.sx(node.t), view.sy(y1)); }
    else { ctx.moveTo(view.sx(x0), view.sy(node.t)); ctx.lineTo(view.sx(x1), view.sy(node.t)); }
    ctx.stroke();
  }

  // Text view of the tree, like a folder listing.
  function treeText(node, prefix, isLast, isRoot) {
    const branch = isRoot ? "" : prefix + (isLast ? "└─ " : "├─ ");
    const line = node.leaf
      ? `${NAMES[node.label]}  (${node.counts[node.label]}/${node.n})`
      : `${AXES[node.f]} < ${node.t.toFixed(2)} ?`;
    if (node.leaf) return branch + line + "\n";
    const childPrefix = isRoot ? "" : prefix + (isLast ? "   " : "│  ");
    return branch + line + "\n" +
      treeText(node.left, childPrefix, false, false) +
      treeText(node.right, childPrefix, true, false);
  }

  function countLeaves(node) { return node.leaf ? 1 : countLeaves(node.left) + countLeaves(node.right); }
  function depthOf(node) { return node.leaf ? 0 : 1 + Math.max(depthOf(node.left), depthOf(node.right)); }

  function render() {
    const { ctx, w: W, h: H } = view;
    ctx.clearRect(0, 0, W, H);
    $("depth-out").textContent = depthInput.value;
    $("leaf-out").textContent = leafInput.value;

    if (points.length) {
      tree = build(points, 0, {
        maxDepth: parseInt(depthInput.value, 10),
        minLeaf: parseInt(leafInput.value, 10),
        impurity: criterionSelect.value === "gini" ? gini : entropy,
      });
      drawNode(tree, BOX, 0);
    } else {
      tree = null;
    }
    drawGrid(view, 1, false);
    for (const p of points) drawPoint(ctx, view.sx(p.x), view.sy(p.y), CLASS_COLORS[p.c], 5);

    $("tree").textContent = tree ? treeText(tree, "", true, true) : "Add some points.";
    $("stat-n").textContent = points.length;
    $("stat-leaves").textContent = tree ? countLeaves(tree) : "–";
    $("stat-depth").textContent = tree ? depthOf(tree) : "–";
    $("stat-acc").textContent = tree
      ? Math.round((100 * points.filter((p) => predict(tree, p) === p.c).length) / points.length) + "%"
      : "–";
  }

  // --- Data -------------------------------------------------------------

  function makeData(kind) {
    const rand = rng({ blobs: 4, xor: 8, circle: 15 }[kind]);
    points = [];
    for (let i = 0; i < 90; i++) {
      let x, y, c;
      if (kind === "blobs") {
        c = i % 2;
        x = (c ? 1.3 : -1.3) + 1.3 * randn(rand);
        y = (c ? 0.8 : -0.8) + 1.3 * randn(rand);
      } else {
        x = -4.6 + 9.2 * rand();
        y = -4.6 + 9.2 * rand();
        c = kind === "xor" ? ((x > 0) !== (y > 0) ? 1 : 0) : (Math.hypot(x, y) < 2.7 ? 1 : 0);
        if (rand() < 0.05) c = 1 - c;            // a little label noise
      }
      points.push({ x: Math.max(-4.8, Math.min(4.8, x)), y: Math.max(-4.8, Math.min(4.8, y)), c });
    }
    render();
  }

  // --- Controls ---------------------------------------------------------

  depthInput.addEventListener("input", render);
  leafInput.addEventListener("input", render);
  criterionSelect.addEventListener("change", render);
  $("blobs").addEventListener("click", () => makeData("blobs"));
  $("xor").addEventListener("click", () => makeData("xor"));
  $("circle").addEventListener("click", () => makeData("circle"));
  $("clear").addEventListener("click", () => { points = []; render(); });

  canvas.addEventListener("pointerdown", (e) => {
    const [sx, sy] = pointerPos(canvas, e);
    const i = hitPoint(points, view, sx, sy);
    if (i >= 0) points.splice(i, 1);
    else points.push({ x: view.wx(sx), y: view.wy(sy), c: +currentClass() });
    render();
  });

  onWidthChange(canvas, () => { view = plot2d(canvas, BOX); render(); });

  view = plot2d(canvas, BOX);
  makeData("circle");
})();
