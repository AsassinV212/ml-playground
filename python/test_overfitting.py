import numpy as np
from numpy.polynomial import legendre

from overfitting import error_curves, features, fit, make_data, mse


def test_features_are_legendre_polynomials():
    x = np.linspace(0, 1, 7)
    assert np.allclose(features(x, 6), legendre.legvander(2 * x - 1, 6))


def test_train_error_never_rises_with_degree():
    # A bigger polynomial family contains the smaller one, so it can always fit at least as well.
    x, y = make_data(15, seed=1)
    train, _ = error_curves(x, y, x, y, max_degree=12)
    assert np.all(np.diff(train) <= 1e-8)


def test_enough_coefficients_pass_through_every_point():
    x, y = make_data(10, seed=2)
    w = fit(x, y, degree=9)                  # 10 coefficients for 10 points
    # Not exactly zero: the tiny 1e-10 stabilizer in fit() matters when weights are in the hundreds.
    assert mse(w, x, y) < 1e-3               # versus noise variance 0.09


def test_test_error_is_u_shaped():
    x, y = make_data(15, seed=1)
    x_test, y_test = make_data(500, seed=99)
    train, test = error_curves(x, y, x_test, y_test, max_degree=14)
    best = int(np.argmin(test))
    assert 3 <= best <= 8                    # a moderate degree wins
    assert test[0] > 2 * test[best]          # underfitting: a constant is too simple
    assert test[14] > 10 * test[best]        # overfitting: the curve chases the noise
    assert train[14] < train[best]           # ...while looking great on training data


def test_l2_shrinks_weights_and_tames_high_degree():
    x, y = make_data(15, seed=1)
    x_test, y_test = make_data(500, seed=99)
    w_free = fit(x, y, 14)
    w_ridge = fit(x, y, 14, l2=1e-3)
    assert np.linalg.norm(w_ridge) < np.linalg.norm(w_free)
    assert mse(w_ridge, x_test, y_test) < mse(w_free, x_test, y_test)
