import os

from auto_heuristic.dataset import load_csv, load_dataset, get_column_values, one_hot_encode_data

def test_load_dataset():
    X, y, feature_names, class_names, encoded_columns = load_dataset(os.path.join("tests", "test_files", "iris.csv"), "target")

    assert X.shape == (150, 4)
    assert y.shape == (150,)
    assert set(feature_names) == {"sepal length", "sepal width", "petal length", "petal width"}
    assert set(class_names) == {"setosa", "versicolor", "virginica"}
    assert encoded_columns == {}


def test_get_column_values():
    data = load_csv(os.path.join("tests", "test_files", "iris.csv"))
    values = get_column_values([j for i in data[1:] for (col, j) in enumerate(i) if col == 4])

    assert values == ["setosa", "versicolor", "virginica"]


def test_one_hot_encode_data():
    data = load_csv(os.path.join("tests", "test_files", "categorical.csv"))
    new_data, encoded_columns = one_hot_encode_data(data, 1)

    assert new_data[0] == ["target", "a", "b", "c"]
    for i in range(1,4):
        column = [row[i] for row in new_data[1:]]
        assert all([v == 1 or v == 0 for v in column])
    assert encoded_columns == {'a': 'category', 'b': 'category', 'c': 'category'}
