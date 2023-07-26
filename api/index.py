import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from auto_heuristic import (
#     load_dataset,
#     get_model,
#     extract_decision_tree,
#     get_variable_list,
#     decision_tree_to_python,
#     decision_tree_to_js,
#     format_return_value,
# )

from flask import Flask, render_template, request
import tempfile

app = Flask(__name__, template_folder="static")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def upload_dataset():
    path = tempfile.NamedTemporaryFile().name
    request.files["file"].save(path)
    target_column = request.values["target"]

    # try:
    #     X, y, feature_names, class_names, encoded_columns = load_dataset(path, target_column)
    #     models = get_model(X, y)
    #     assert models, "No successful heuristic found"
    #     return_format = format_return_value(set(y))
    #     options = []

    #     for depth, (clf, score) in models.items():
    #         formatted_tree = extract_decision_tree(clf, feature_names, class_names, encoded_columns)
    #         variable_list = get_variable_list(formatted_tree)
    #         python_code = decision_tree_to_python(formatted_tree, variable_list, return_format, score)
    #         js_code = decision_tree_to_js(formatted_tree, variable_list, return_format, score)
    #         options.append({"depth": depth, "score": score, "python_code": python_code, "js_code": js_code})
    # except AssertionError as e:
    #     return render_template("index.html", error=e)

    return render_template("index.html", options=reversed([]))
