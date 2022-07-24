from auto_heuristic.code import (
    decision_tree_to_js,
    decision_tree_to_python,
    decision_tree_to_text,
    format_return_value,
    get_variable_list,
)
from auto_heuristic.tree import DecisionNode
import os

CHILD_TRUE_BRANCH = DecisionNode("petal width", 1.65, "versicolor", "virginica")
CHILD_FALSE_BRANCH = DecisionNode("petal width", 1.75, "versicolor", "virginica")
FALSE_BRANCH = DecisionNode("petal length", 4.75, CHILD_TRUE_BRANCH, CHILD_FALSE_BRANCH)
DECISION_TREE = DecisionNode("petal length", 2.45, "setosa", FALSE_BRANCH)
FEATURE_NAMES = ["petal width", "petal length"]
RETURN_FORMAT = {"setosa": '"setosa"', "versicolor": '"versicolor"', "virginica": '"virginica"'}
PERFORMANCE = 1.0


def test_get_variable_list():
    assert set(get_variable_list(DECISION_TREE)) == set(FEATURE_NAMES)


def test_decision_tree_to_text():
    with open(os.path.join("tests", "test_files", "sample.txt")) as f:
        text = f.read()
    assert decision_tree_to_text(DECISION_TREE) + "\n" == text


def test_decision_tree_to_python():
    with open(os.path.join("tests", "test_files", "sample.py")) as f:
        python_code = f.read()
    assert decision_tree_to_python(DECISION_TREE, FEATURE_NAMES, RETURN_FORMAT, PERFORMANCE) + "\n" == python_code


def test_decision_tree_to_js():
    with open(os.path.join("tests", "test_files", "sample.js")) as f:
        js_code = f.read()
    assert decision_tree_to_js(DECISION_TREE, FEATURE_NAMES, RETURN_FORMAT, PERFORMANCE) + "\n" == js_code


def test_format_return_value():
    assert format_return_value({"0", "1"}) == {"0": False, "1": True}
    assert format_return_value({"1", "2", "3"}) == {"1": 1, "2": 2, "3": 3}
    assert format_return_value({"1", "1.5", "2"}) == {"1": 1.0, "1.5": 1.5, "2": 2.0}
    assert format_return_value({"a", "b", "c"}) == {"a": '"a"', "b": '"b"', "c": '"c"'}
