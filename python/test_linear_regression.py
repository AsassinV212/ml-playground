import numpy as np
import pytest

from linear_regression import (
    fit_gradient_descent, fit_least_squares, mae, mae_grad, mse, mse_grad,
)


@pytest.fixture
def noisy_line():
    rand = np.random.default_rng(1)
    x = rand.uniform(-1, 1, 30)
    y = 0.6 * x + 0.2 + rand.normal(0, 0.05, 30)
    return x, y


def test_least_squares_recovers_exact_line():
    x = np.array([-1.0, 0.0, 0.5, 2.0])
    w, b = fit_least_squares(x, 3 * x - 1)
    assert w == pytest.approx(3)
    assert b == pytest.approx(-1)


def test_least_squares_matches_numpy_polyfit(noisy_line):
    x, y = noisy_line
    assert np.allclose(fit_least_squares(x, y), np.polyfit(x, y, 1))


def test_gradient_descent_reaches_least_squares(noisy_line):
    x, y = noisy_line
    assert np.allclose(fit_gradient_descent(x, y, "mse", 0.3, 2000), fit_least_squares(x, y), atol=1e-6)


@pytest.mark.parametrize("loss, grad", [(mse, mse_grad), (mae, mae_grad)])
def test_gradients_match_finite_differences(noisy_line, loss, grad):
    x, y = noisy_line
    w, b, h = 0.3, -0.4, 1e-6
    gw = (loss(x, y, w + h, b) - loss(x, y, w - h, b)) / (2 * h)
    gb = (loss(x, y, w, b + h) - loss(x, y, w, b - h)) / (2 * h)
    assert np.allclose(grad(x, y, w, b), (gw, gb), atol=1e-5)


def test_mae_resists_an_outlier_better_than_mse(noisy_line):
    x, y = noisy_line
    x_out = np.append(x, 0.9)
    y_out = np.append(y, -3.0)                 # one point far below the line
    w_mse, _ = fit_least_squares(x_out, y_out)
    w_mae, _ = fit_gradient_descent(x_out, y_out, "mae", 0.1, 5000)
    assert abs(w_mae - 0.6) < 0.1                      # MAE barely notices
    assert abs(w_mse - 0.6) > 3 * abs(w_mae - 0.6)     # MSE is dragged towards the outlier
