from auto_heuristic import (
    load_dataset,
    get_model,
    extract_decision_tree,
    get_variable_list,
    decision_tree_to_python,
    decision_tree_to_js,
    format_return_value,
)
import argparse


def auto_heuristic(csv_path: str, target_column: str, python_path: str = None, js_path: str = None):
    X, y, feature_names, class_names = load_dataset(csv_path, target_column)
    models = get_model(X, y)
    assert models, "No successful heuristic found"
    return_format = format_return_value(set(y))
    depth = max(models)
    model, score = models[depth]

    print("Best depth:", depth)
    print("Score:", score)

    formatted_tree = extract_decision_tree(model, feature_names, class_names)
    variable_list = get_variable_list(formatted_tree)

    if python_path:
        python_code = decision_tree_to_python(formatted_tree, variable_list, return_format, score)
        with open(python_path, "w") as f:
            f.write(python_code + "\n")

    if js_path:
        js_code = decision_tree_to_js(formatted_tree, variable_list, return_format, score)
        with open(js_path, "w") as f:
            f.write(js_code + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate heuristic rules for a given CSV")
    parser.add_argument("file", metavar="N", type=str, help="CSV file to process")
    parser.add_argument("--target", type=str, required=True, help="Column to target")
    parser.add_argument("--python", type=str, required=False, help="Python file to generate")
    parser.add_argument("--js", type=str, required=False, help="JS file to generate")

    args = parser.parse_args()

    auto_heuristic(args.file, args.target, args.python, args.js)
