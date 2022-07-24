from auto_heuristic import (
    load_dataset,
    get_model,
    extract_decision_tree,
    get_variable_list,
    decision_tree_to_python,
    decision_tree_to_js,
)

from flask import Flask, render_template, request

app = Flask(__name__, template_folder="static")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def upload_dataset():
    request.files["file"].save("temp.csv")
    target_column = request.values["target"]
    X, y, feature_names, class_names = load_dataset("temp.csv", target_column)
    models = get_model(X, y)
    assert models, "No successful heuristic found"
    options = []

    for depth, (clf, score) in models.items():
        formatted_tree = extract_decision_tree(clf, feature_names, class_names)
        variable_list = get_variable_list(formatted_tree)
        python_code = decision_tree_to_python(formatted_tree, variable_list)
        js_code = decision_tree_to_js(formatted_tree, variable_list)
        options.append({"depth": depth, "score": score, "python_code": python_code, "js_code": js_code})

    return render_template("index.html", options=options)


if __name__ == "__main__":
    app.run(debug=True)
