"""Logistic regression for two classes, trained by gradient descent.

Matches topics/logistic-regression.js. Labels are 0 and 1;
the model outputs p = P(label = 1 | x) = sigmoid(w . x + b).
"""

import numpy as np


def sigmoid(z):
    """1 / (1 + e^-z), written so that large |z| never overflows."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1 + ez)
    return out


def predict_proba(X, w, b):
    return sigmoid(X @ w + b)


def log_loss(X, t, w, b, l2=0.0):
    """Average cross-entropy plus an optional L2 penalty (l2 / 2) * |w|^2 (bias not penalized)."""
    p = np.clip(predict_proba(X, w, b), 1e-12, 1 - 1e-12)
    return np.mean(-t * np.log(p) - (1 - t) * np.log(1 - p)) + 0.5 * l2 * np.dot(w, w)


def log_loss_grad(X, t, w, b, l2=0.0):
    # The gradient has the same simple shape as for linear regression:
    # (prediction - target) times the input.
    err = predict_proba(X, w, b) - t
    return X.T @ err / len(t) + l2 * w, np.mean(err)


def fit(X, t, learning_rate=0.3, n_steps=1000, l2=0.0):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(n_steps):
        gw, gb = log_loss_grad(X, t, w, b, l2)
        w = w - learning_rate * gw
        b = b - learning_rate * gb
    return w, b


def accuracy(X, t, w, b):
    return np.mean((predict_proba(X, w, b) >= 0.5) == t)


if __name__ == "__main__":
    rand = np.random.default_rng(0)
    X = np.vstack([rand.normal([-1.5, -1], 1, (30, 2)), rand.normal([1.5, 1], 1, (30, 2))])
    t = np.repeat([0, 1], 30)
    w, b = fit(X, t, l2=0.01)
    print("w =", w, "b =", b, "accuracy =", accuracy(X, t, w, b))
