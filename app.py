from sympy import python
from auto_heuristic import (
    load_iris_dataset,
    get_model,
    extract_decision_tree,
    get_variable_list,
    decision_tree_to_python,
    decision_tree_to_js,
    decision_tree_to_text,
)

from flask import Flask, render_template
app = Flask(__name__, template_folder="static")

@app.route("/")
def home():
    X, y, feature_names, class_names = load_iris_dataset()
    models = get_model(X, y)
    options = []

    for depth, (clf, score) in models.items():
        formatted_tree = extract_decision_tree(clf, feature_names, class_names)
        variable_list = get_variable_list(formatted_tree)
        python_code = decision_tree_to_python(formatted_tree, variable_list)
        js_code = decision_tree_to_js(formatted_tree, variable_list)
        options.append({
            "depth": depth,
            "score": score,
            "python_code": python_code,
            "js_code": js_code
        })
    
    return render_template("index.html", options=options)

if __name__ == '__main__':
    app.run(debug =True)
