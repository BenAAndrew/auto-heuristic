from sklearn import datasets
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from typing import List, Tuple


def get_column_values(column: pd.Series) -> List[str]:
    target_values = []
    for value in column:
        if value not in target_values:
            target_values.append(value)

    return target_values


def load_dataset(path: str, target_column: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], dict]:
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
            column_categories = get_column_values(df[column])
            for cat in column_categories:
                col = [1 if df[column][j] == cat else 0 for j in range(len(df[column]))]
                df[cat] = col
                encoded_columns[cat] = column

            df = df.drop(columns=[column])

    data_columns = [c for c in list(df.columns) if c != target_column]
    target_values = get_column_values(df[target_column])

    return np.array(df[data_columns]), np.array(df[target_column]), data_columns, target_values, encoded_columns
