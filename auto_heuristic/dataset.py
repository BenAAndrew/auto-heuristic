from sklearn import datasets
import numpy as np
from typing import List, Tuple


def load_iris_dataset() -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    iris = datasets.load_iris()
    return iris.data, iris.target, [i[:-5] for i in iris.feature_names], list(iris.target_names)
