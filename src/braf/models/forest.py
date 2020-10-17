from random import randrange

from braf.models.tree import RandomDecisionTreeClassifier
from braf.utils.knn import get_k_nearest_neighbors
from braf.utils.evaluation import split_by_label

import logging

log = logging.getLogger(__name__)


class RandomForestClassifier(RandomDecisionTreeClassifier):

    def __init__(self, max_depth=10, min_size=1, n_features=1, sample_size=1, n_trees=1, **kwargs):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.forest = None
        self.trained = False
        super().__init__(max_depth, min_size, n_features, **kwargs)

    def __add__(self, other):
        """
        Sum two RF classifiers so that the result is a RF classifier with:
            - n_trees is the sum of the values of n_trees of each instance we are summing
            - forest is the concatenation of the forest of each instance we are summing
        """
        self.forest = self.forest + other.forest
        self.n_trees = self.n_trees + other.n_trees
        return self

    @staticmethod
    def _subsample(data, ratio):
        """
        Create random subsample from the dataset with replacement
        """
        sample = list()
        n_sample = round(len(data) * ratio)
        while len(sample) < n_sample:
            index = randrange(len(data))
            sample.append(data[index])
        return sample

    def _bagging_proba(self, row):
        """
        Estimate probability with a list of bagged trees
        """
        predictions = [tree.predict(row) for tree in self.forest]
        return sum(predictions)/len(predictions)

    def _bagging_predict(self, row):
        """
        If probability > 0.5, predict 1
        """
        return int(self._bagging_proba(row) > 0.5)

    def train(self, data):
        """
        Random Forest Algorithm
        For each i in 0 to n_trees:
            - get random subsample of training data
            - train a random tree on it
            - add tree to forest
        Note: DecisionTreeClassifier instances are based on a random subset of n_features features
        """
        forest = list()
        for i in range(self.n_trees):
            sample = self._subsample(data, self.sample_size)
            tree = RandomDecisionTreeClassifier(self.max_depth, self.min_size, self.n_features)
            tree.train(sample)
            forest.append(tree)
        self.forest = forest
        self.trained = True
        return self

    def predict(self, data):
        if not self.trained:
            log.info("Model is not trained.")
            return
        predictions = [self._bagging_predict(row) for row in data]
        return predictions

    def predict_probabilities(self, data):
        if not self.trained:
            log.info("Model is not trained.")
            return
        predictions = [self._bagging_proba(row) for row in data]
        return predictions


class BiasedRandomForestClassifier(RandomForestClassifier):

    def __init__(self, p=0.5, k=5, max_depth=10, min_size=1, n_features=1, sample_size=1, n_trees=1, **kwargs):
        self.p = p
        self.k = k
        self.forest = None
        super().__init__(max_depth, min_size, n_features, sample_size, n_trees, **kwargs)

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        if not 0 <= value <= 1:
            raise ValueError("p must be between 0 and 1.")
        self._p = value

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, value):
        if not value > 0:
            raise ValueError("k must be > 0.")
        self._k = value

    def _build_critical(self, data):
        t_maj, t_min = split_by_label(data)
        t_c = []
        for t_i in t_min:
            t_c.append(t_i)  # Adds the ti to the critical data set.
            t_nn = get_k_nearest_neighbors(t_maj, t_i, self.k)  # Find knn for each minority instance in data set.
            for t_j in t_nn:
                if t_j not in t_c:
                    t_c.append(t_j)  # Add the unique neighbors only to the critical data set.
        return t_c

    def train(self, data):
        """
        Biased Random Forest Classifier is built here based on algorithm from the paper.
        """
        log.info("Training Biased Random Forest Classifier.")
        # Build critical set
        t_c = self._build_critical(data)
        # Set number of trees to be trained on critical vs regular  set
        n_trees_1 = round(self.n_trees * (1 - self.p) - 0.5)
        n_trees_2 = round(self.n_trees * self.p + 0.5)
        # Build a first forest based on the full data set of size S×(1−p).
        rf1 = RandomForestClassifier(self.max_depth, self.min_size, self.n_features, self.sample_size, n_trees_1)
        rf1.train(data)
        # Build a second forest based on the critical data set data set of size S×p.
        rf2 = RandomForestClassifier(self.max_depth, self.min_size, self.n_features, self.sample_size, n_trees_2)
        rf2.train(t_c)
        # Combine the two forests to generate the main forest RF.
        braf = rf1 + rf2
        self.forest = braf.forest
        self.trained = True
        return braf
