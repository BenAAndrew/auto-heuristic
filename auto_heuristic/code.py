from typing import List
from auto_heuristic.tree import DecisionNode


def get_variable_list(tree: DecisionNode) -> List[str]:
    variables = [tree.condition_var]
    if isinstance(tree.true_decision, DecisionNode):
        variables.extend(get_variable_list(tree.true_decision))
    if isinstance(tree.false_decision, DecisionNode):
        variables.extend(get_variable_list(tree.false_decision))
    return list(set(variables))


def format_return_value(return_values: set) -> dict:
    def _is_float(element: any) -> bool:
        try:
            float(element)
            return True
        except ValueError:
            return False

    if return_values == {"0", "1"}:
        return {"0": False, "1": True}
    elif all([_is_float(i) for i in return_values]):
        if all([i.isnumeric() for i in return_values]):
            return {k: int(k) for k in return_values}
        else:
            return {k: float(k) for k in return_values}
    else:
        return {k: f'"{k}"' for k in return_values}


def decision_tree_to_python(
    tree: DecisionNode, feature_names: List[str], return_format: dict, performance: float
) -> str:
    variable_names = {f: f.replace(" ", "_").lower() for f in feature_names}
    code = "def predict({}):".format(", ".join(variable_names.values())) + "\n"
    code += "    # Accuracy: {}%".format(int(performance * 100)) + "\n"

    def _decision_node_to_python(node, depth=1):
        indent = "    " * depth
        if isinstance(node, str):
            return "{}return {}".format(indent, return_format[node])
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


def decision_tree_to_js(tree: DecisionNode, feature_names: List[str], return_format: dict, performance: float) -> str:
    variable_names = {
        f: "".join([w.lower() if i == 0 else w.capitalize() for i, w in enumerate(f.replace("_", " ").split(" "))])
        for f in feature_names
    }
    code = "function predict({}) {{".format(", ".join(variable_names.values())) + "\n"
    code += "  // Accuracy: {}%".format(int(performance * 100)) + "\n"

    def _decision_node_to_js(node, depth=1):
        indent = "  " * depth
        if isinstance(node, str):
            return_value = return_format[node]
            # convert bool to lowercase
            if isinstance(return_value, bool):
                return_value = "true" if return_value else "false"
            return "{}return {};".format(indent, return_value)
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


def decision_tree_to_text(tree: DecisionNode) -> str:
    def _decision_node_to_text(node, depth=0):
        indent = "|   " * depth + "|--- "
        if isinstance(node, str):
            return "{}class: {}".format(indent, node)
        else:
            return (
                "{}{} <= {}".format(indent, node.condition_var, node.condition_value)
                + "\n"
                + _decision_node_to_text(node.true_decision, depth + 1)
                + "\n"
                + "{}{} > {}".format(indent, node.condition_var, node.condition_value)
                + "\n"
                + _decision_node_to_text(node.false_decision, depth + 1)
            )

    return _decision_node_to_text(tree)
