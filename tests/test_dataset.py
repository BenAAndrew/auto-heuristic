from auto_heuristic.dataset import load_dataset, convert_column
import os
import pandas as pd


def test_load_dataset():
    X, y, feature_names, class_names = load_dataset(os.path.join("tests", "test_files", "iris.csv"), "target")

    assert X.shape == (150, 4)
    assert y.shape == (150,)
    assert set(feature_names) == {"sepal length", "sepal width", "petal length", "petal width"}
    assert set(class_names) == {"setosa", "versicolor", "virginica"}


def test_convert_column():
    df = pd.read_csv(os.path.join("tests", "test_files", "iris.csv"))
    column_categories, converted_column = convert_column(df[["target"]])

    assert set(column_categories) == {"setosa", "versicolor", "virginica"}
    assert converted_column.shape == (150, 3)
