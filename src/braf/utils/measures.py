import math


def compute_gini_index(groups, classes):
    """
    Compute the gini index for a split data set
    """
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini


def compute_euclidean_distance(row1, row2):
    """
    Compute the Euclidean distance between two vectors
    """
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return math.sqrt(distance)
