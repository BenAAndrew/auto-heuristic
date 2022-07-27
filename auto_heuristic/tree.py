from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass
class DecisionNode:
    condition_var: str
    condition_value: float
    true_decision: any
    false_decision: any


def extract_decision_tree(
    clf: DecisionTreeClassifier, feature_names: List[str], class_names: List[str], encoded_columns: dict
) -> DecisionNode:
    tree = clf.tree_
    tree_feature_names = [feature_names[i] if i != _tree.TREE_UNDEFINED else None for i in tree.feature]

    def recurse(node, depth):
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            variable = tree_feature_names[node]
            if variable in encoded_columns:
                # Categorical field
                return DecisionNode(
                    condition_var=encoded_columns[variable],
                    condition_value=variable,
                    true_decision=recurse(tree.children_right[node], depth + 1),
                    false_decision=recurse(tree.children_left[node], depth + 1),
                )
            else:
                # Continuous field
                return DecisionNode(
                    condition_var=variable,
                    condition_value=np.round(tree.threshold[node], 2),
                    true_decision=recurse(tree.children_left[node], depth + 1),
                    false_decision=recurse(tree.children_right[node], depth + 1),
                )
        else:
            return class_names[np.argmax(tree.value[node])]

    return recurse(0, 1)
