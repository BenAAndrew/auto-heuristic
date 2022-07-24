from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from typing import List, Tuple


def load_iris_dataset() -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    iris = datasets.load_iris()
    return iris.data, iris.target, [i[:-5] for i in iris.feature_names], list(iris.target_names)


def convert_column(column: pd.DataFrame) -> Tuple[np.array, np.array]:
    enc = OneHotEncoder()
    transformed = enc.fit_transform(column)
    return enc.categories_[0], transformed.toarray()


def load_dataset(path: str, target_column: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    df = pd.read_csv(path)
    assert df.notnull().values.all(), "Invalid CSV. Has null values"
    columns = list(df.columns)
    assert len(columns) > 1, "Invalid CSV. Only one column"
    assert target_column in columns, f"{target_column} not in list of columns ({columns})"
    assert len(df) > 10, "Insufficent samples in CSV"
    df[target_column] = df[target_column].astype(str)

    # One-hot encode non-numeric columns
    encoded_columns = {}
    for column in columns:
        if column != target_column and not is_numeric_dtype(df[column]):
            column_categories, converted_column = convert_column(df[[column]])
            encoded_columns[column] = column_categories
            df[column] = converted_column

    data_columns = [c for c in columns if c != target_column]

    target_values = []
    for value in df[target_column]:
        if value not in target_values:
            target_values.append(value)

    return np.array(df[data_columns]), np.array(df[target_column]), data_columns, target_values
