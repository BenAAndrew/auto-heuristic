import os
from sklearn.model_selection import train_test_split

from auto_heuristic.dataset import load_dataset
from auto_heuristic.model import get_model
from auto_heuristic.tree import DecisionNode, extract_decision_tree
from tests.test_files.iris import predict as continuous_predict
from tests.test_files.categorical import predict as categorical_predict


def test_extract_decision_tree():
    X, y, feature_names, class_names, _ = load_dataset(os.path.join("tests", "test_files", "iris.csv"), "target")
    models = get_model(X, y)
    latest_model = models[3][0]
    tree = extract_decision_tree(latest_model, feature_names, class_names, {})

    assert tree.condition_var == "petal length"
    assert tree.condition_value == 2.45
    assert tree.true_decision == "setosa"
    assert isinstance(tree.false_decision, DecisionNode)

    assert tree.false_decision.condition_var == "petal length"
    assert tree.false_decision.condition_value == 4.75
    assert isinstance(tree.false_decision.true_decision, DecisionNode)
    assert isinstance(tree.false_decision.false_decision, DecisionNode)

    assert tree.false_decision.true_decision.condition_var == "petal width"
    assert tree.false_decision.true_decision.condition_value == 1.65
    assert tree.false_decision.true_decision.true_decision == "versicolor"
    assert tree.false_decision.true_decision.false_decision == "virginica"

    assert tree.false_decision.false_decision.condition_var == "petal width"
    assert tree.false_decision.false_decision.condition_value == 1.75
    assert tree.false_decision.false_decision.true_decision == "versicolor"
    assert tree.false_decision.false_decision.false_decision == "virginica"


def test_continuous_tree_decision_quality():
    X, y, _, _, _ = load_dataset(os.path.join("tests", "test_files", "iris.csv"), "target")
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for i, sample in enumerate(X_test):
        _, _, petal_length, petal_width = sample
        prediction = continuous_predict(petal_width, petal_length)
        assert prediction == y_test[i]


def test_categorical_tree_decision_quality():
    X, y, data_columns, _, _ = load_dataset(os.path.join("tests", "test_files", "categorical.csv"), "target")
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for i, sample in enumerate(X_test):
        index = list(sample).index(1)
        prediction = categorical_predict(data_columns[index])
        assert prediction == y_test[i]
