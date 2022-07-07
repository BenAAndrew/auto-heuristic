from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np

DEPTH_RANGE = range(1, 6)
MIN_SCORE = 0.5


def get_model(X: np.ndarray, y: np.ndarray) -> DecisionTreeClassifier:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {}
    last_score = 0

    for depth in DEPTH_RANGE:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=1234)
        model = clf.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        # Stop iterating if performance is not improving
        if score <= last_score:
            break
        elif score >= MIN_SCORE:
            models[depth] = (model, np.round(score, 2))
        last_score = score

    return models
