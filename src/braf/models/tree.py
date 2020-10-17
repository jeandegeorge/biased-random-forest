import sys
from abc import ABC, abstractmethod
from random import randrange

from braf.utils.measures import compute_gini_index
from braf.constants import AppConstants

import logging

log = logging.getLogger(__name__)


class DecisionTreeClassifier(ABC):
    """
    This class defines a standard decision tree. Could be implemented in the future but out of scope in current project.
    """

    def __init__(self, max_depth=10, min_size=1, **kwargs):
        self.max_depth = max_depth  # max depth of tree
        self.min_size = min_size  # minimum size of a split
        self.root = None  # root of tree
        self.trained = False  # True if tree instance is trained
        self._features = None

    @staticmethod
    def _to_terminal(group):
        """
        Returns most frequent label in group of observations
        """
        labels = [row[-1] for row in group]
        return max(labels, key=labels.count)

    @staticmethod
    def _split_by_value(index, value, data):
        """
        Split a data set based on an attribute value
        """
        left, right = list(), list()
        for row in data:
            try:
                if row[index] < value:
                    left.append(row)
                else:
                    right.append(row)
            except TypeError as e:
                log.fatal(str(e) + '. Please check all input data is numerical.')
                sys.exit(1)
        return left, right

    def _find_best_cutoff(self, data, features):
        class_values = list(set(row[-1] for row in data))
        best_index, best_value, best_score, best_groups = 999, 999, 999, None
        for feature in features:
            for row in data:
                groups = self._split_by_value(feature, row[feature], data)
                gini = compute_gini_index(groups, class_values)
                if gini < best_score:
                    best_index, best_value, best_score, best_groups = feature, row[feature], gini, groups
        return {AppConstants.INDEX: best_index, AppConstants.VALUE: best_value, AppConstants.GROUPS: best_groups}

    @abstractmethod
    def _get_split(self, data):
        """
        Out of scope, to be implemented.
        Will contain method to build stndard decision tree based on greedy algorithm.
        """
        pass

    def _split(self, node, depth):
        """
        Create child splits for a node or make terminal
        """
        left, right = node[AppConstants.GROUPS]
        del(node[AppConstants.GROUPS])
        # If no split
        if not left or not right:
            node[AppConstants.LEFT] = node[AppConstants.RIGHT] = self._to_terminal(left + right)
            return
        # If max depth over limit
        if depth >= self.max_depth:
            node[AppConstants.LEFT], node[AppConstants.RIGHT] = self._to_terminal(left), self._to_terminal(right)
            return
        # Split left if not too small
        if len(left) <= self.min_size:
            node[AppConstants.LEFT] = self._to_terminal(left)
        else:
            node[AppConstants.LEFT] = self._get_split(left)
            self._split(node[AppConstants.LEFT], depth + 1)
        # Split right if not too small
        if len(right) <= self.min_size:
            node[AppConstants.RIGHT] = self._to_terminal(right)
        else:
            node[AppConstants.RIGHT] = self._get_split(right)
            self._split(node[AppConstants.RIGHT], depth + 1)

    def _predict(self, node, row):
        """
        Make a prediction at any node
        """
        if row[node[AppConstants.INDEX]] < node[AppConstants.VALUE]:
            if isinstance(node[AppConstants.LEFT], dict):
                return self._predict(node[AppConstants.LEFT], row)
            else:
                return node[AppConstants.LEFT]
        else:
            if isinstance(node[AppConstants.RIGHT], dict):
                return self._predict(node[AppConstants.RIGHT], row)
            else:
                return node[AppConstants.RIGHT]

    def train(self, data):
        """
        Build decision tree
        """
        self._features = [n for n, _ in enumerate(data[0][:-1])]
        self.root = self._get_split(data)
        self._split(self.root, 1)
        self.trained = True
        return self

    def predict(self, row):
        """
        Return prediction with a decision tree
        """
        if not self.trained:
            log.info("Model is not trained.")
            return

        return self._predict(node=self.root, row=row)


class RandomDecisionTreeClassifier(DecisionTreeClassifier):
    """
    This class defines a decision tree built on "n_features" features chosen at random from the total features.
    """

    def __init__(self, max_depth=10, min_size=1, n_features=1, **kwargs):
        self.n_features = n_features  # number of features to pick at random
        super().__init__(max_depth, min_size, **kwargs)

    def _get_split(self, data):
        """
        Select the best split point for a dataset at random
        """
        features = list()
        # Build subset of features randomly
        while len(features) < self.n_features:
            index = randrange(len(data[0])-1)
            if index not in features:
                features.append(index)
        return self._find_best_cutoff(data, features)
