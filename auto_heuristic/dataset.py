import numpy as np
import csv
from typing import List, Tuple


def get_column_values(column: List[str]) -> List[str]:
    target_values = []
    for value in column:
        if value not in target_values:
            target_values.append(value)

    return target_values


def column_is_numeric(column: List[str]) -> bool:
    return all([i.replace(".", "").isnumeric() for i in column])


def one_hot_encode_data(dataset: List[List[str]], target_column_index: int) -> Tuple[List[List[str]]]:
    n_columns = len(dataset[0])
    new_columns = {}
    encoded_columns = {}
    columns_to_delete = []

    for i in range(n_columns):
        if i != target_column_index:
            column_name = dataset[0][i]
            values = [row[i] for row in dataset[1:]]
            if not column_is_numeric(values):
                column_values = get_column_values(values)
                for value in column_values:
                    col = [1 if v == value else 0 for v in values]
                    new_columns[value] = col
                    encoded_columns[value] = column_name
                columns_to_delete.append(i)
    
    for column in columns_to_delete[::-1]:
        for row in dataset:
            del row[column]

    for column_name, values in new_columns.items():
        dataset[0].append(column_name)
        for i, row in enumerate(dataset[1:]):
            row.append(values[i])
    
    return dataset, encoded_columns


def load_csv(path: str) -> List[List[str]]:
    with open(path) as csv_file:
        dataset = list(csv.reader(csv_file, delimiter=','))
        n_columns = len(dataset[0])
        assert n_columns > 1, "Invalid CSV. Only one column"
        assert len(dataset) > 10, "Insufficent samples in CSV"
        for row in dataset:
            assert len(row) == n_columns, "Invalid CSV. Has null values"
            for value in row:
                assert value is not None and value != "", "Invalid CSV. Has null values"
    return dataset


def load_dataset(path: str, target_column: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], dict]:
    csv = load_csv(path)
    try:
        target_column_index = csv[0].index(target_column)
    except:
        raise Exception(f"{target_column} not in list of columns ({header})")

    csv, encoded_columns = one_hot_encode_data(csv, target_column_index)
    header = csv[0]
    data = csv[1:]
    target_column_index = csv[0].index(target_column)
    x = np.array([[float(j) for (col, j) in enumerate(i) if col != target_column_index] for i in data])
    y = np.array([j for i in data for (col, j) in enumerate(i) if col == target_column_index])
    data_columns = [i for (col,i) in enumerate(header) if col != target_column_index]
    target_values = get_column_values(y)

    return x, y, data_columns, target_values, encoded_columns
