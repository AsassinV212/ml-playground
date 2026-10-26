"""Gradient descent on 2D loss surfaces: a readable reference implementation.

The surfaces and the update rule here match topics/gradient-descent.js,
so you can reproduce anything you see on the page.

Run it directly to print a short trace:

    python python/gradient_descent.py
"""

import numpy as np


# ---------------------------------------------------------------------------
# Loss surfaces. Each one has a loss function f(p) and its gradient grad_f(p),
# where p = [x, y] is a point in parameter space.
# ---------------------------------------------------------------------------

def bowl(p):
    """Round bowl: f = x^2 + y^2. One minimum at (0, 0)."""
    x, y = p
    return x**2 + y**2


def bowl_grad(p):
    x, y = p
    return np.array([2 * x, 2 * y])


def valley(p):
    """Stretched valley: f = 0.5 x^2 + 5 y^2. Steep in y, shallow in x."""
    x, y = p
    return 0.5 * x**2 + 5 * y**2


def valley_grad(p):
    x, y = p
    return np.array([x, 10 * y])


def local_min(p):
    """Tilted double well: a deep minimum near x = -1.48, a shallow one near x = 1.33."""
    x, y = p
    return 0.25 * x**4 - x**2 + 0.3 * x + 0.5 * y**2


def local_min_grad(p):
    x, y = p
    return np.array([x**3 - 2 * x + 0.3, y])


SURFACES = {
    "bowl": (bowl, bowl_grad),
    "valley": (valley, valley_grad),
    "local_min": (local_min, local_min_grad),
}


# ---------------------------------------------------------------------------
# The algorithm.
# ---------------------------------------------------------------------------

def gradient_descent(grad_f, start, learning_rate, n_steps, tol=1e-8):
    """Run plain gradient descent and return the path it takes.

    Parameters
    ----------
    grad_f : callable
        Function returning the gradient (a numpy array) at a point.
    start : sequence of float
        Starting point.
    learning_rate : float
        Step size: how far to move along the negative gradient each step.
    n_steps : int
        Maximum number of steps.
    tol : float
        Stop early once the gradient is shorter than this (we are at a flat spot).

    Returns
    -------
    numpy.ndarray of shape (k + 1, dim)
        Every point visited, starting with `start`.
    """
    p = np.asarray(start, dtype=float)
    path = [p.copy()]
    for _ in range(n_steps):
        g = grad_f(p)                  # which way is uphill, and how steep
        if np.linalg.norm(g) < tol:
            break
        p = p - learning_rate * g      # take a step downhill
        path.append(p.copy())
    return np.array(path)


if __name__ == "__main__":
    f, grad_f = SURFACES["valley"]
    path = gradient_descent(grad_f, start=[-2.5, 1.0], learning_rate=0.15, n_steps=10)
    for i, p in enumerate(path):
        print(f"step {i:2d}  x={p[0]:+.4f}  y={p[1]:+.4f}  loss={f(p):.5f}")
