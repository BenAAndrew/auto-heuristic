from auto_heuristic.code import (
    decision_tree_to_js,
    decision_tree_to_python,
    format_return_value,
    get_variable_list,
    format_if_statement,
)
from auto_heuristic.tree import DecisionNode
import os

PERFORMANCE = 1.0
RETURN_FORMAT = {"setosa": '"setosa"', "versicolor": '"versicolor"', "virginica": '"virginica"'}

# Continuous tree
CONTINUOUS_CHILD_TRUE_BRANCH = DecisionNode("petal width", 1.65, "versicolor", "virginica")
CONTINUOUS_CHILD_FALSE_BRANCH = DecisionNode("petal width", 1.75, "versicolor", "virginica")
CONTINUOUS_FALSE_BRANCH = DecisionNode(
    "petal length", 4.75, CONTINUOUS_CHILD_TRUE_BRANCH, CONTINUOUS_CHILD_FALSE_BRANCH
)
CONTINUOUS_DECISION_TREE = DecisionNode("petal length", 2.45, "setosa", CONTINUOUS_FALSE_BRANCH)
CONTINUOUS_FEATURE_NAMES = ["petal width", "petal length"]

# Categorical tree
CATEGORICAL_FALSE_BRANCH = DecisionNode("category", "a", "setosa", "versicolor")
CATEGORICAL_DECISION_TREE = DecisionNode("category", "c", "virginica", CATEGORICAL_FALSE_BRANCH)
CATEGORICAL_FEATURE_NAMES = ["category"]


def test_get_variable_list():
    assert set(get_variable_list(CONTINUOUS_DECISION_TREE)) == set(CONTINUOUS_FEATURE_NAMES)


def test_format_return_value():
    assert format_return_value({"0", "1"}) == {"0": False, "1": True}
    assert format_return_value({"1", "2", "3"}) == {"1": 1, "2": 2, "3": 3}
    assert format_return_value({"1", "1.5", "2"}) == {"1": 1.0, "1.5": 1.5, "2": 2.0}
    assert format_return_value({"a", "b", "c"}) == {"a": '"a"', "b": '"b"', "c": '"c"'}


def test_format_if_statement():
    assert format_if_statement("a", "b") == 'a == "b"'
    assert format_if_statement("a", 1) == "a <= 1"


def test_continuous_decision_tree_to_python():
    with open(os.path.join("tests", "test_files", "iris.py")) as f:
        python_code = f.read()
    assert (
        decision_tree_to_python(CONTINUOUS_DECISION_TREE, CONTINUOUS_FEATURE_NAMES, RETURN_FORMAT, PERFORMANCE) + "\n"
        == python_code
    )


def test_continuous_decision_tree_to_js():
    with open(os.path.join("tests", "test_files", "iris.js")) as f:
        js_code = f.read()
    assert (
        decision_tree_to_js(CONTINUOUS_DECISION_TREE, CONTINUOUS_FEATURE_NAMES, RETURN_FORMAT, PERFORMANCE) + "\n"
        == js_code
    )


def test_categorical_decision_tree_to_python():
    with open(os.path.join("tests", "test_files", "categorical.py")) as f:
        python_code = f.read()
    assert (
        decision_tree_to_python(CATEGORICAL_DECISION_TREE, CATEGORICAL_FEATURE_NAMES, RETURN_FORMAT, PERFORMANCE) + "\n"
        == python_code
    )


def test_categorical_decision_tree_to_js():
    with open(os.path.join("tests", "test_files", "categorical.js")) as f:
        js_code = f.read()
    assert (
        decision_tree_to_js(CATEGORICAL_DECISION_TREE, CATEGORICAL_FEATURE_NAMES, RETURN_FORMAT, PERFORMANCE) + "\n"
        == js_code
    )
