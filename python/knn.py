"""k-nearest neighbors classification: a readable reference implementation.

Matches topics/knn.js, including how ties are broken: if several classes get
the same number of votes, the one whose member is closest to the query wins.
"""

import numpy as np


def distances(X, q, metric="euclidean"):
    """Distance from every row of X to the point q."""
    diff = X - q
    if metric == "manhattan":
        return np.sum(np.abs(diff), axis=1)   # |dx| + |dy|: walking along a street grid
    return np.sqrt(np.sum(diff**2, axis=1))   # straight-line distance


def predict_one(X, y, q, k, metric="euclidean"):
    d = distances(X, q, metric)
    nearest = np.argsort(d, kind="stable")[:k]     # indices of the k closest points
    votes = np.bincount(y[nearest])
    winners = np.flatnonzero(votes == votes.max())
    for i in nearest:                              # tie-break: closest point among the winners
        if y[i] in winners:
            return y[i]


def predict(X, y, Q, k, metric="euclidean"):
    return np.array([predict_one(X, y, q, k, metric) for q in Q])


def leave_one_out_accuracy(X, y, k, metric="euclidean"):
    """Classify each point using all the *other* points. An honest estimate of accuracy."""
    hits = 0
    for i in range(len(X)):
        rest = np.arange(len(X)) != i
        hits += predict_one(X[rest], y[rest], X[i], k, metric) == y[i]
    return hits / len(X)


if __name__ == "__main__":
    rand = np.random.default_rng(0)
    X = np.vstack([rand.normal(c, 1.0, (20, 2)) for c in ([-2, -1.5], [2, -1.5], [0, 2.2])])
    y = np.repeat([0, 1, 2], 20)
    for k in (1, 5, 15):
        print(f"k={k:2d}  leave-one-out accuracy = {leave_one_out_accuracy(X, y, k):.2f}")
