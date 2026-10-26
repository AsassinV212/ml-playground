import numpy as np
import pytest

from logistic_regression import accuracy, fit, log_loss, log_loss_grad, sigmoid


def blobs(gap, seed=0):
    rand = np.random.default_rng(seed)
    X = np.vstack([rand.normal([-gap, 0], 0.5, (25, 2)), rand.normal([gap, 0], 0.5, (25, 2))])
    return X, np.repeat([0.0, 1.0], 25)


def test_sigmoid_basics():
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert sigmoid(2.0) + sigmoid(-2.0) == pytest.approx(1.0)
    big = sigmoid(np.array([-1000.0, 1000.0]))     # no overflow warnings, no NaN
    assert np.allclose(big, [0.0, 1.0])


@pytest.mark.parametrize("l2", [0.0, 0.1])
def test_gradient_matches_finite_differences(l2):
    X, t = blobs(1.0)
    w, b, h = np.array([0.3, -0.7]), 0.2, 1e-6
    gw, gb = log_loss_grad(X, t, w, b, l2)
    for i in range(2):
        d = np.zeros(2)
        d[i] = h
        num = (log_loss(X, t, w + d, b, l2) - log_loss(X, t, w - d, b, l2)) / (2 * h)
        assert gw[i] == pytest.approx(num, abs=1e-6)
    num_b = (log_loss(X, t, w, b + h, l2) - log_loss(X, t, w, b - h, l2)) / (2 * h)
    assert gb == pytest.approx(num_b, abs=1e-6)


def test_separable_data_is_classified_perfectly():
    X, t = blobs(3.0)
    w, b = fit(X, t)
    assert accuracy(X, t, w, b) == 1.0


def test_without_l2_weights_keep_growing_on_separable_data():
    # The sigmoid can always get a bit sharper and the loss a bit lower.
    # The growth is slow (roughly like log of the step count) but never stops.
    X, t = blobs(3.0)
    norms = [np.linalg.norm(fit(X, t, n_steps=n)[0]) for n in (500, 5000, 20000)]
    assert norms[0] + 0.5 < norms[1] < norms[2] - 0.3


def test_l2_keeps_weights_bounded():
    X, t = blobs(3.0)
    w_free, _ = fit(X, t, n_steps=5000)
    w_reg, _ = fit(X, t, n_steps=5000, l2=0.1)
    w_reg_longer, _ = fit(X, t, n_steps=10000, l2=0.1)
    assert np.linalg.norm(w_reg) < np.linalg.norm(w_free)
    assert np.allclose(w_reg, w_reg_longer, atol=1e-3)   # it has settled
