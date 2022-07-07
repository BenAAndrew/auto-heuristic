from auto_heuristic.dataset import load_dataset
import os


def test_load_dataset():
    X, y, feature_names, class_names = load_dataset(os.path.join("tests", "test_files", "iris.csv"), "target")

    assert X.shape == (150, 4)
    assert y.shape == (150,)
    assert set(feature_names) == {"sepal length", "sepal width", "petal length", "petal width"}
    assert set(class_names) == {"setosa", "versicolor", "virginica"}
