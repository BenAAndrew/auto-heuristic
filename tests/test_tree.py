from auto_heuristic.dataset import load_iris_dataset
from auto_heuristic.model import get_model
from auto_heuristic.tree import DecisionNode, extract_decision_tree


def test_extract_decision_tree():
    X, y, feature_names, class_names = load_iris_dataset()
    model = get_model(X, y)
    tree = extract_decision_tree(model, feature_names, class_names)

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
