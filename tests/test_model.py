from auto_heuristic.dataset import load_iris_dataset
from auto_heuristic.model import get_model


def test_get_model():
    X, y, _, _ = load_iris_dataset()
    model = get_model(X, y)
    assert model.score(X, y) >= 0.95
