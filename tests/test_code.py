from auto_heuristic.code import decision_tree_to_js, decision_tree_to_python, decision_tree_to_text, get_variable_list
from auto_heuristic.tree import DecisionNode
import os

CHILD_TRUE_BRANCH = DecisionNode("petal width", 1.65, "versicolor", "virginica")
CHILD_FALSE_BRANCH = DecisionNode("petal width", 1.75, "versicolor", "virginica")
FALSE_BRANCH = DecisionNode("petal length", 4.75, CHILD_TRUE_BRANCH, CHILD_FALSE_BRANCH)
DECISION_TREE = DecisionNode("petal length", 2.45, "setosa", FALSE_BRANCH)
FEATURE_NAMES = ["petal width", "petal length"]


def test_get_variable_list():
    assert set(get_variable_list(DECISION_TREE)) == set(FEATURE_NAMES)


def test_decision_tree_to_text():
    with open(os.path.join("tests", "test_files", "sample.txt")) as f:
        text = f.read()
    assert decision_tree_to_text(DECISION_TREE) + "\n" == text


def test_decision_tree_to_python():
    with open(os.path.join("tests", "test_files", "sample.py")) as f:
        python_code = f.read()
    assert decision_tree_to_python(DECISION_TREE, FEATURE_NAMES) + "\n" == python_code


def test_decision_tree_to_js():
    with open(os.path.join("tests", "test_files", "sample.js")) as f:
        js_code = f.read()
    assert decision_tree_to_js(DECISION_TREE, FEATURE_NAMES) + "\n" == js_code
