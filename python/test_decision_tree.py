import numpy as np
import pytest

from decision_tree import build_tree, depth, entropy, gini, leaves, predict


def test_impurity_values():
    assert gini(np.array([5, 0])) == pytest.approx(0.0)
    assert gini(np.array([5, 5])) == pytest.approx(0.5)
    assert entropy(np.array([5, 0])) == pytest.approx(0.0)
    assert entropy(np.array([5, 5])) == pytest.approx(1.0)     # one bit: a fair coin
    assert entropy(np.array([1, 1, 1, 1])) == pytest.approx(2.0)


def test_pure_node_is_a_leaf():
    X = np.array([[0.0, 1.0], [2.0, 3.0]])
    tree = build_tree(X, np.array([1, 1]))
    assert tree["leaf"] and tree["label"] == 1


def test_single_split_with_threshold_between_points():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    y = np.array([0, 0, 1, 1])
    tree = build_tree(X, y, max_depth=5)
    assert tree["feature"] == 0 and tree["threshold"] == pytest.approx(2.0)
    assert len(leaves(tree)) == 2


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_deep_tree_memorizes_training_data(criterion):
    rand = np.random.default_rng(0)
    X = rand.uniform(-1, 1, (60, 2))
    y = rand.integers(0, 2, 60)                  # pure noise, still fitted perfectly
    tree = build_tree(X, y, max_depth=30, criterion=criterion)
    assert np.all(predict(tree, X) == y)


def test_limits_are_respected():
    rand = np.random.default_rng(1)
    X = rand.uniform(-1, 1, (80, 2))
    y = rand.integers(0, 2, 80)
    tree = build_tree(X, y, max_depth=3, min_leaf=5)
    assert depth(tree) <= 3
    assert all(leaf["counts"].sum() >= 5 for leaf in leaves(tree))


def test_greedy_gives_up_on_perfect_xor():
    # Any single cut leaves both halves exactly 50/50, so no first split helps,
    # even though two splits would separate the classes perfectly.
    X = np.array([[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]])
    y = np.array([0, 0, 1, 1])
    assert build_tree(X, y, max_depth=5)["leaf"]


def test_two_levels_solve_a_lopsided_xor():
    X = np.array([[-1.0, -1.0]] * 6 + [[1.0, 1.0]] * 4 + [[-1.0, 1.0]] * 4 + [[1.0, -1.0]] * 4)
    y = np.array([0] * 10 + [1] * 8)
    tree = build_tree(X, y, max_depth=2)
    assert np.all(predict(tree, X) == y)
