from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from typing import List


class DecisionNode:
    condition_var: str
    condition_value: float
    true_decision: any
    false_decision: any

    def __init__(self, condition_var, condition_value, true_decision, false_decision):
        self.condition_var = condition_var
        self.condition_value = np.round(condition_value, 2)
        self.true_decision = true_decision
        self.false_decision = false_decision

    def to_dict(self) -> dict:
        return {
            "var": self.condition_var,
            "threshold": self.condition_value,
            "true_decision": self.true_decision.to_dict()
            if isinstance(self.true_decision, DecisionNode)
            else self.true_decision,
            "false_decision": self.false_decision.to_dict()
            if isinstance(self.false_decision, DecisionNode)
            else self.false_decision,
        }


def extract_decision_tree(
    tree: DecisionTreeClassifier, feature_names: List[str], class_names: List[str]
) -> DecisionNode:
    tree_ = tree.tree_
    tree_feature_names = [feature_names[i] if i != _tree.TREE_UNDEFINED else None for i in tree_.feature]

    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            return DecisionNode(
                tree_feature_names[node],
                tree_.threshold[node],
                recurse(tree_.children_left[node], depth + 1),
                recurse(tree_.children_right[node], depth + 1),
            )
        else:
            return class_names[np.argmax(tree_.value[node])]

    return recurse(0, 1)
