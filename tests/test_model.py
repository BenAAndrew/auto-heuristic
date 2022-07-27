import os

from auto_heuristic.dataset import load_dataset
from auto_heuristic.model import MIN_SCORE, get_model


def test_get_model():
    X, y, _, _, _ = load_dataset(os.path.join("tests", "test_files", "iris.csv"), "target")
    models = get_model(X, y)
    assert list(models.keys()) == [1, 2, 3]

    for (_, score) in models.values():
        assert score >= MIN_SCORE

    # Last model score
    assert models[3][1] == 1
