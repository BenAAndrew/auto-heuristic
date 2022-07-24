from auto_heuristic import (
    load_dataset,
    get_model,
    extract_decision_tree,
    decision_tree_to_python,
    decision_tree_to_js,
    format_return_value,
    get_variable_list,
)
import os

target_column = "target"
file_path = os.path.join("tests", "test_files", "iris.csv")
X, y, feature_names, class_names = load_dataset(file_path, target_column)
models = get_model(X, y)
assert models, "No models generated"
options = []
return_format = format_return_value(set(y))
formatted_tree = None
variable_list = None

for depth, (clf, score) in models.items():
    formatted_tree = extract_decision_tree(clf, feature_names, class_names)
    variable_list = get_variable_list(formatted_tree)
    print(f"{depth}: {score*100}%")

with open("sample.py", "w") as f:
    f.write(decision_tree_to_python(formatted_tree, variable_list, return_format) + "\n")

with open("sample.js", "w", newline="") as f:
    f.write(decision_tree_to_js(formatted_tree, variable_list, return_format) + "\n")
