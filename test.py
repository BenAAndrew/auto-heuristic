from auto_heuristic import (
    load_dataset,
    get_model,
    extract_decision_tree,
    get_variable_list,
    decision_tree_to_python,
    decision_tree_to_js,
)
import os

target_column = "target"
file_path = os.path.join("tests", "test_files", "iris.csv")
X, y, feature_names, class_names = load_dataset(file_path, target_column)
models = get_model(X, y)
assert models, "No models generated"
options = []
formatted_tree = None

for depth, (clf, score) in models.items():
    formatted_tree = extract_decision_tree(clf, feature_names, class_names)
    variable_list = get_variable_list(formatted_tree)
    python_code = decision_tree_to_python(formatted_tree, variable_list)
    js_code = decision_tree_to_js(formatted_tree, variable_list)
    print(f"{depth}: {score*100}%")

with open("sample.py", "w") as f:
    f.write(decision_tree_to_python(formatted_tree, feature_names) + "\n")

with open("sample.js", "w", newline="") as f:
    f.write(decision_tree_to_js(formatted_tree, feature_names) + "\n")
