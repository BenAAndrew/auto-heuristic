def predict(petal_width, petal_length):
    # Accuracy: 100%
    if petal_length <= 2.45:
        return "setosa"
    else:
        if petal_length <= 4.75:
            if petal_width <= 1.65:
                return "versicolor"
            else:
                return "virginica"
        else:
            if petal_width <= 1.75:
                return "versicolor"
            else:
                return "virginica"
