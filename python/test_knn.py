import numpy as np

from knn import distances, leave_one_out_accuracy, predict, predict_one


def test_distances():
    X = np.array([[3.0, 4.0], [0.0, 0.0]])
    assert np.allclose(distances(X, np.zeros(2)), [5.0, 0.0])
    assert np.allclose(distances(X, np.zeros(2), "manhattan"), [7.0, 0.0])


def test_k1_memorizes_training_data():
    rand = np.random.default_rng(0)
    X = rand.normal(size=(40, 2))
    y = rand.integers(0, 3, 40)                       # even random labels
    assert np.all(predict(X, y, X, k=1) == y)


def test_majority_vote():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [1.1, 0.0], [1.2, 0.0]])
    y = np.array([0, 1, 1, 1])
    q = np.array([0.1, 0.0])
    assert predict_one(X, y, q, k=1) == 0             # the nearest point alone
    assert predict_one(X, y, q, k=3) == 1             # outvoted two to one


def test_tie_goes_to_the_closest_class():
    X = np.array([[1.0, 0.0], [2.0, 0.0]])
    y = np.array([1, 0])
    assert predict_one(X, y, np.zeros(2), k=2) == 1


def test_metric_changes_the_answer():
    # A is diagonal, B is straight ahead. Euclid: A is closer. Manhattan: B is closer.
    X = np.array([[1.0, 1.0], [1.6, 0.0]])
    y = np.array([0, 1])
    q = np.zeros(2)
    assert predict_one(X, y, q, 1, "euclidean") == 0
    assert predict_one(X, y, q, 1, "manhattan") == 1


def test_larger_k_ignores_label_noise():
    rand = np.random.default_rng(3)
    X = np.vstack([rand.normal([-2, 0], 1.0, (40, 2)), rand.normal([2, 0], 1.0, (40, 2))])
    y = np.repeat([0, 1], 40)
    flip = rand.choice(80, 12, replace=False)         # 15% of the labels are wrong
    y[flip] = 1 - y[flip]
    assert leave_one_out_accuracy(X, y, 9) > leave_one_out_accuracy(X, y, 1)
