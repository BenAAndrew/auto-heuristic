from sklearn.model_selection import train_test_split
from auto_heuristic.dataset import load_iris_dataset
from auto_heuristic.model import get_model
from auto_heuristic.tree import DecisionNode, extract_decision_tree
from tests.test_files.sample import predict


def test_extract_decision_tree():
    X, y, feature_names, class_names = load_iris_dataset()
    models = get_model(X, y)
    latest_model = models[3][0]
    tree = extract_decision_tree(latest_model, feature_names, class_names)

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


def test_tree_decision_quality():
    X, y, _, class_names = load_iris_dataset()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for i, sample in enumerate(X_test):
        _, _, petal_length, petal_width = sample
        prediction = predict(petal_width, petal_length)
        prediction_index = class_names.index(prediction)
        assert prediction_index == y_test[i]
