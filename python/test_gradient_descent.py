import numpy as np
import pytest

from gradient_descent import SURFACES, gradient_descent


@pytest.mark.parametrize("name", SURFACES)
def test_gradient_matches_finite_differences(name):
    f, grad_f = SURFACES[name]
    h = 1e-6
    for p in [np.array([0.7, -1.2]), np.array([-1.9, 0.4]), np.array([2.2, 1.1])]:
        numeric = np.array([
            (f(p + [h, 0]) - f(p - [h, 0])) / (2 * h),
            (f(p + [0, h]) - f(p - [0, h])) / (2 * h),
        ])
        assert np.allclose(grad_f(p), numeric, atol=1e-5)


def test_converges_to_bowl_minimum():
    _, grad_f = SURFACES["bowl"]
    path = gradient_descent(grad_f, start=[2.0, -1.5], learning_rate=0.1, n_steps=200)
    assert np.allclose(path[-1], [0.0, 0.0], atol=1e-6)


def test_loss_never_increases_with_small_learning_rate():
    f, grad_f = SURFACES["valley"]
    path = gradient_descent(grad_f, start=[-2.5, 1.0], learning_rate=0.05, n_steps=100)
    losses = [f(p) for p in path]
    assert all(b <= a for a, b in zip(losses, losses[1:]))


def test_tiny_learning_rate_is_slow():
    f, grad_f = SURFACES["bowl"]
    slow = gradient_descent(grad_f, [2.0, 2.0], learning_rate=0.005, n_steps=20)
    fast = gradient_descent(grad_f, [2.0, 2.0], learning_rate=0.1, n_steps=20)
    assert f(slow[-1]) > 100 * f(fast[-1])


def test_too_large_learning_rate_diverges():
    # On the bowl each step multiplies the point by (1 - 2 * lr).
    # With lr = 1.1 that factor is -1.2: the ball overshoots further every time.
    f, grad_f = SURFACES["bowl"]
    path = gradient_descent(grad_f, [1.0, 1.0], learning_rate=1.1, n_steps=20)
    assert f(path[-1]) > 100 * f(path[0])


def test_valley_zigzags_across_steep_direction():
    # y is 10x steeper than x, so a learning rate close to the limit 0.2
    # makes y flip sign every step while x creeps along slowly.
    _, grad_f = SURFACES["valley"]
    path = gradient_descent(grad_f, [-2.5, 1.0], learning_rate=0.18, n_steps=6)
    y_signs = np.sign(path[:, 1])
    assert all(a != b for a, b in zip(y_signs, y_signs[1:]))


def test_start_point_decides_which_minimum():
    f, grad_f = SURFACES["local_min"]
    right = gradient_descent(grad_f, [2.0, 1.0], learning_rate=0.05, n_steps=2000)[-1]
    left = gradient_descent(grad_f, [-2.0, 1.0], learning_rate=0.05, n_steps=2000)[-1]
    assert right[0] == pytest.approx(1.33, abs=0.01)   # stuck in the shallow well
    assert left[0] == pytest.approx(-1.48, abs=0.01)   # found the deep well
    assert f(left) < f(right)
