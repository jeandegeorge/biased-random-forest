from braf.utils.measures import *


def get_k_nearest_neighbors(train, ref, k):
    distances = list()
    for train_row in train:
        dist = compute_euclidean_distance(ref, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors
