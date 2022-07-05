from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np


def get_model(X: np.ndarray, y: np.ndarray) -> DecisionTreeClassifier:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the classifier with max_depth=3
    clf = DecisionTreeClassifier(max_depth=3, random_state=1234)
    model = clf.fit(X_train, y_train)
    # print(clf.score(X_test, y_test))
    return model
