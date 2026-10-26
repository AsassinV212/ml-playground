"""Linear regression y ~ w * x + b: exact least squares and gradient descent.

Matches topics/linear-regression.js. One feature keeps the formulas short;
the same ideas work with a matrix X and a weight vector w.
"""

import numpy as np


def predict(x, w, b):
    return w * x + b


def mse(x, y, w, b):
    """Mean squared error: average of squared residuals."""
    r = predict(x, w, b) - y
    return np.mean(r**2)


def mae(x, y, w, b):
    """Mean absolute error: average of absolute residuals."""
    r = predict(x, w, b) - y
    return np.mean(np.abs(r))


def mse_grad(x, y, w, b):
    r = predict(x, w, b) - y                  # residuals: how far each prediction misses
    return 2 * np.mean(r * x), 2 * np.mean(r)


def mae_grad(x, y, w, b):
    # |r| has slope -1 or +1, so every point pulls with the same strength.
    s = np.sign(predict(x, w, b) - y)
    return np.mean(s * x), np.mean(s)


def fit_least_squares(x, y):
    """Exact minimum of MSE: slope = cov(x, y) / var(x), line passes through the mean point."""
    x_mean, y_mean = np.mean(x), np.mean(y)
    w = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    b = y_mean - w * x_mean
    return w, b


def fit_gradient_descent(x, y, loss="mse", learning_rate=0.1, n_steps=1000):
    """Fit w, b by gradient descent on MSE or MAE.

    For MAE the step shrinks over time (learning_rate / sqrt(1 + t / 100)):
    its gradient never gets small near the minimum, so a fixed step would jitter forever.
    """
    grad = mse_grad if loss == "mse" else mae_grad
    w, b = 0.0, 0.0
    for t in range(n_steps):
        gw, gb = grad(x, y, w, b)
        lr = learning_rate if loss == "mse" else learning_rate / np.sqrt(1 + t / 100)
        w -= lr * gw
        b -= lr * gb
    return w, b


if __name__ == "__main__":
    rand = np.random.default_rng(0)
    x = rand.uniform(-1, 1, 20)
    y = 0.6 * x + 0.2 + rand.normal(0, 0.1, 20)
    print("least squares   :", fit_least_squares(x, y))
    print("gradient descent:", fit_gradient_descent(x, y))
