from csv import reader
from braf.utils.knn import *


def load_csv(file):
    data = list()
    with open(file, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            data.append(row)
    return data


def str_column_to_float(data, column):
    """
    Convert string values in data to float
    """
    for row in data:
        row[column] = float(row[column].strip())


def str_column_to_int(data, column):
    """
    Convert string values in data to integer
    """
    class_values = [row[column] for row in data]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in data:
        row[column] = lookup[row[column]]
    return lookup


def impute(data, columns, k):
    for n, row in enumerate(data):
        if n == 0:
            continue
        elif not all([row[c] for c in columns]):
            for c in columns:
                row_candidates = [r for i, r in enumerate(data) if i not in (0, n) and r[c] != 0]
                neighbors = get_k_nearest_neighbors(row_candidates, row, k)
                if row[c] == 0:
                    impute_value = sum([n[c] for n in neighbors])/k
                    row[c] = impute_value
