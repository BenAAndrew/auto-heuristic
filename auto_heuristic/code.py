from typing import List
from auto_heuristic.tree import DecisionNode


def get_variable_list(tree: DecisionNode) -> List[str]:
    variables = [tree.condition_var]
    if isinstance(tree.true_decision, DecisionNode):
        variables.extend(get_variable_list(tree.true_decision))
    if isinstance(tree.false_decision, DecisionNode):
        variables.extend(get_variable_list(tree.false_decision))
    return list(set(variables))


def decision_tree_to_python(tree: DecisionNode, feature_names: List[str]) -> str:
    variable_names = {f: f.replace(" ", "_").lower() for f in feature_names}
    code = ""
    code += "def predict({}):".format(", ".join(variable_names.values())) + "\n"

    def _decision_node_to_python(node, depth=1):
        indent = "    " * depth
        if isinstance(node, str):
            return '{}return "{}"'.format(indent, node)
        else:
            return (
                "{}if {} <= {}:".format(indent, variable_names[node.condition_var], node.condition_value)
                + "\n"
                + _decision_node_to_python(node.true_decision, depth + 1)
                + "\n"
                + "{}else:".format(indent)
                + "\n"
                + _decision_node_to_python(node.false_decision, depth + 1)
            )

    code += _decision_node_to_python(tree)
    return code


def decision_tree_to_js(tree: DecisionNode, feature_names: List[str]) -> str:
    variable_names = {
        f: "".join([w.lower() if i == 0 else w.capitalize() for i, w in enumerate(f.replace("_", " ").split(" "))])
        for f in feature_names
    }
    code = ""
    code += "function predict({}) {{".format(", ".join(variable_names.values())) + "\n"

    def _decision_node_to_js(node, depth=1):
        indent = "  " * depth
        if isinstance(node, str):
            return '{}return "{}";'.format(indent, node)
        else:
            return (
                "{}if ({} <= {}) {{".format(indent, variable_names[node.condition_var], node.condition_value)
                + "\n"
                + _decision_node_to_js(node.true_decision, depth + 1)
                + "\n"
                + "{}}} else {{".format(indent)
                + "\n"
                + _decision_node_to_js(node.false_decision, depth + 1)
                + "\n"
                + "{}}}".format(indent)
            )

    code += _decision_node_to_js(tree)
    code += "\n}"
    return code
