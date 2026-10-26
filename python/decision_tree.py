"""A decision tree classifier built greedily, one best split at a time.

Matches topics/decision-tree.js: same impurity measures, same split search,
same tie-breaking (first feature, then smallest threshold; majority ties go
to the smaller class index).
"""

import numpy as np


def gini(counts):
    """Chance that two random picks from the node have different labels."""
    p = counts / counts.sum()
    return 1 - np.sum(p**2)


def entropy(counts):
    """Shannon entropy in bits: how many yes/no questions are still needed on average."""
    p = counts / counts.sum()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


IMPURITY = {"gini": gini, "entropy": entropy}


def best_split(X, y, n_classes, criterion="gini", min_leaf=1):
    """Return (feature, threshold) with the lowest weighted child impurity, or None."""
    impurity = IMPURITY[criterion]
    n = len(y)
    parent = impurity(np.bincount(y, minlength=n_classes))
    best, best_score = None, parent - 1e-12            # a split must actually help
    for f in range(X.shape[1]):
        order = np.argsort(X[:, f], kind="stable")
        xs, ys = X[order, f], y[order]
        left = np.zeros(n_classes)
        total = np.bincount(ys, minlength=n_classes).astype(float)
        for i in range(1, n):                          # left = first i points
            left[ys[i - 1]] += 1
            if i < min_leaf or n - i < min_leaf or xs[i - 1] == xs[i]:
                continue
            right = total - left
            score = (i * impurity(left) + (n - i) * impurity(right)) / n
            if score < best_score:
                best, best_score = (f, (xs[i - 1] + xs[i]) / 2), score
    return best


def build_tree(X, y, n_classes=None, max_depth=3, min_leaf=1, criterion="gini", depth=0):
    """Grow the tree recursively. A node is a dict: a leaf or a split with two children."""
    if n_classes is None:
        n_classes = int(y.max()) + 1
    counts = np.bincount(y, minlength=n_classes)
    leaf = {"leaf": True, "label": int(np.argmax(counts)), "counts": counts}
    if depth >= max_depth or np.count_nonzero(counts) <= 1:
        return leaf
    split = best_split(X, y, n_classes, criterion, min_leaf)
    if split is None:
        return leaf
    f, t = split
    go_left = X[:, f] < t
    return {
        "leaf": False, "feature": f, "threshold": t,
        "left": build_tree(X[go_left], y[go_left], n_classes, max_depth, min_leaf, criterion, depth + 1),
        "right": build_tree(X[~go_left], y[~go_left], n_classes, max_depth, min_leaf, criterion, depth + 1),
    }


def predict_one(node, x):
    while not node["leaf"]:
        node = node["left"] if x[node["feature"]] < node["threshold"] else node["right"]
    return node["label"]


def predict(node, X):
    return np.array([predict_one(node, x) for x in X])


def leaves(node):
    return [node] if node["leaf"] else leaves(node["left"]) + leaves(node["right"])


def depth(node):
    return 0 if node["leaf"] else 1 + max(depth(node["left"]), depth(node["right"]))


if __name__ == "__main__":
    rand = np.random.default_rng(0)
    X = rand.uniform(-4, 4, (120, 2))
    y = (np.hypot(X[:, 0], X[:, 1]) < 2.5).astype(int)     # class 1 inside a circle
    for d in (1, 2, 4, 8):
        tree = build_tree(X, y, max_depth=d)
        acc = np.mean(predict(tree, X) == y)
        print(f"max_depth={d}: {len(leaves(tree)):2d} leaves, training accuracy {acc:.2f}")
