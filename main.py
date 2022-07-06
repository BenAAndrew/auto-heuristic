from auto_heuristic import (
    load_iris_dataset,
    get_model,
    extract_decision_tree,
    get_variable_list,
    decision_tree_to_python,
    decision_tree_to_js,
    decision_tree_to_text,
)

X, y, feature_names, class_names = load_iris_dataset()
clf = get_model(X, y)
formatted_tree = extract_decision_tree(clf, feature_names, class_names)
feature_names = get_variable_list(formatted_tree)

with open("sample.py", "w") as f:
    f.write(decision_tree_to_python(formatted_tree, feature_names) + "\n")

with open("sample.js", "w", newline="") as f:
    f.write(decision_tree_to_js(formatted_tree, feature_names) + "\n")

with open("sample.txt", "w", newline="") as f:
    f.write(decision_tree_to_text(formatted_tree) + "\n")
