from sklearn import datasets
import numpy as np
import pandas as pd
from typing import List, Tuple


def load_iris_dataset() -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    iris = datasets.load_iris()
    return iris.data, iris.target, [i[:-5] for i in iris.feature_names], list(iris.target_names)


def load_dataset(path: str, target_column: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    df = pd.read_csv(path)
    columns = list(df.columns)
    assert target_column in columns, f"{target_column} not in list of columns ({columns})"
    data_columns = [c for c in columns if c != target_column]
    target_values = list(set(df[target_column]))
    return np.array(df[data_columns]), np.array(df[target_column]), data_columns, target_values
