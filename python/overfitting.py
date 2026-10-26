"""Overfitting with polynomial regression: train error vs test error.

Matches topics/overfitting.js. We fit polynomials of growing degree to a few
noisy samples of sin(2 pi x) and compare the error on the training points with
the error on fresh test points.

Polynomials are built from Legendre polynomials of t = 2x - 1 instead of plain
powers x^k: the same curves, but the linear system is far better conditioned.
"""

import numpy as np


def true_function(x):
    return np.sin(2 * np.pi * x)


def make_data(n, noise=0.3, seed=0):
    rand = np.random.default_rng(seed)
    x = rand.uniform(0, 1, n)
    return x, true_function(x) + noise * rand.normal(size=n)


def features(x, degree):
    """Columns P_0(t) ... P_degree(t), Legendre polynomials of t = 2x - 1."""
    t = 2 * np.asarray(x, dtype=float) - 1
    cols = [np.ones_like(t), t]
    for k in range(1, degree):
        cols.append(((2 * k + 1) * t * cols[k] - k * cols[k - 1]) / (k + 1))
    return np.stack(cols[: degree + 1], axis=1)


def fit(x, y, degree, l2=0.0):
    """Minimize mean squared error + l2 * |w|^2 (ridge regression), solved exactly.

    A tiny extra 1e-10 keeps the system solvable when there are more
    coefficients than points.
    """
    phi = features(x, degree)
    n = len(y)
    A = phi.T @ phi / n + (l2 + 1e-10) * np.eye(degree + 1)
    return np.linalg.solve(A, phi.T @ y / n)


def predict(w, x):
    return features(x, len(w) - 1) @ w


def mse(w, x, y):
    return np.mean((predict(w, x) - y) ** 2)


def error_curves(x_train, y_train, x_test, y_test, max_degree=15, l2=0.0):
    """Train and test MSE for every degree from 0 to max_degree."""
    train, test = [], []
    for d in range(max_degree + 1):
        w = fit(x_train, y_train, d, l2)
        train.append(mse(w, x_train, y_train))
        test.append(mse(w, x_test, y_test))
    return np.array(train), np.array(test)


if __name__ == "__main__":
    x, y = make_data(15, seed=1)
    x_test, y_test = make_data(500, seed=99)
    train, test = error_curves(x, y, x_test, y_test)
    for d, (a, b) in enumerate(zip(train, test)):
        print(f"degree {d:2d}: train {a:.4f}   test {b:10.4f}")
