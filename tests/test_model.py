from auto_heuristic.dataset import load_iris_dataset
from auto_heuristic.model import MIN_SCORE, get_model


def test_get_model():
    X, y, _, _ = load_iris_dataset()
    models = get_model(X, y)
    assert list(models.keys()) == [1,2,3]

    for (clf, score) in models.values():
        assert score >= MIN_SCORE

    # Last model score
    assert models[3][1] >= 0.95
