import pytest
from math import isclose

from braf.utils.evaluation import compute_auprc, compute_accuracy, compute_auroc, compute_recall, compute_precision


def test_compute_auprc():
    y_true = [0, 1, 0, 1]
    y_proba = [0.6, 0.6, 0.6, 0.6]
    assert isclose(compute_auprc(y_true, y_proba), 0.75, abs_tol=1e-8)


def test_compute_auroc():
    y_true = [0, 1, 0, 1]
    y_proba = [0.6, 0.6, 0.6, 0.6]
    assert isclose(compute_auroc(y_true, y_proba), 0.5, abs_tol=1e-8)


def test_compute_recall():
    y_true = [0, 1, 0, 1]
    y_proba = [1, 1, 1, 1]
    assert isclose(compute_recall(y_true, y_proba), 1, abs_tol=1e-8)


def test_compute_precision():
    y_true = [0, 1, 0, 1]
    y_proba = [1, 1, 1, 1]
    assert isclose(compute_precision(y_true, y_proba), 0.5, abs_tol=1e-8)


def test_compute_accuracy():
    y_true = [0, 1, 0, 1]
    y_proba = [1, 1, 1, 1]
    assert isclose(compute_accuracy(y_true, y_proba), 0.5, abs_tol=1e-8)
