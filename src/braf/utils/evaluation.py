import os
import matplotlib.pyplot as plt
from random import randrange, shuffle


def generate_output(actual, predicted, probabilities, message=None, suffix=""):
    """
    Generate output required in task instructions
    """
    accuracy = compute_accuracy(actual, predicted)
    recall = compute_recall(actual, predicted)
    precision = compute_precision(actual, predicted)
    auprc = compute_auprc(actual, probabilities)
    auroc = compute_auroc(actual, probabilities)
    print(message)
    print("___")
    print("Accuracy = " + str(accuracy))
    print("Recall = " + str(recall))
    print("Precision = " + str(precision))
    print("AUPRC score = " + str(auprc))
    print("AUROC score = " + str(auroc))
    print("___")
    build_roc_curve(actual, probabilities, suffix)
    build_prc_curve(actual, probabilities, suffix)
    return accuracy, recall, precision, auprc, auroc


def run_cross_validation(data, model, n_folds):
    """
    Evaluate an algorithm using a cross validation split
    """
    folds = cross_validation_split(data, n_folds)
    scores = list()
    n = 0
    for fold in folds:
        n += 1
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        model.train(train_set)
        actual = [row[-1] for row in fold]
        predicted = model.predict(test_set)
        probabilities = model.predict_probabilities(test_set)
        message = "Fold-%s results:" % n
        suffix = "_fold%s" % n
        scores.append(generate_output(actual, predicted, probabilities, message, suffix))
    print("Average Cross Validation Result:")
    print("___")
    print("Average accuracy: " + str(round(sum([s[0] for s in scores])/len(scores),2)))
    print("Average recall: " + str(round(sum([s[1] for s in scores])/len(scores),2)))
    print("Average precision: " + str(round(sum([s[2] for s in scores])/len(scores),2)))
    print("Average AUPRC: " + str(round(sum([s[3] for s in scores])/len(scores),2)))
    print("Average AUROC: " + str(round(sum([s[4] for s in scores])/len(scores),2)))
    print("___")
    return scores


def compute_accuracy(actual, predicted):
    """
    Compute accuracy
    """
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return round(correct / float(len(actual)), 2)


def cross_validation_split(data, folds=3):
    """
    Split a data set into k folds
    """
    data_split = []
    data_copy = list(data)
    fold_size = int(len(data) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(data_copy))
            fold.append(data_copy.pop(index))
        data_split.append(fold)
    return data_split


def compute_precision(y_true, y_pred):
    if sum(y_pred) == 0:
        return
    tp = 0
    fp = 0
    for i, j in zip(y_true, y_pred):
        tp += all((i, j))
        fp += (i == 0 and j == 1)
    return round(tp / (tp + fp), 2)


def compute_recall(y_true, y_pred):
    if sum(y_true) == 0:
        return
    tp = 0
    fn = 0
    for i, j in zip(y_true, y_pred):
        tp += all((i, j))
        fn += (i == 1 and j == 0)
    return round(tp / (tp + fn), 2)


def compute_fpr(y_true, y_pred):
    if all(y_true):
        return
    fp = 0
    tn = 0
    for i, j in zip(y_true, y_pred):
        fp += (i == 0 and j == 1)
        tn += not any((i, j))
    return round(fp / (fp + tn), 2)


def compute_auprc(y_true, y_proba):
    thresholds = [0.1 * i for i in range(-1, 12)]
    y_preds = [[int(p > t) for p in y_proba] for t in thresholds]
    recall_scores = [compute_recall(y_true, y_pred) for y_pred in y_preds]
    recall_scores = list(map(lambda x: x if x is not None else 0, recall_scores))
    precision_scores = [compute_precision(y_true, y_pred) for y_pred in y_preds]
    precision_scores = list(map(lambda x: x if x is not None else 1, precision_scores))
    score_pairs = list(zip(recall_scores, precision_scores))
    score_pairs = sorted(score_pairs, key=lambda x: (x[0], x[1]))
    auprc = 0
    for i in range(1, len(score_pairs)):
        try:
            auprc += (score_pairs[i][0] - score_pairs[i-1][0]) * (score_pairs[i][1] + score_pairs[i-1][1]) / 2
        except TypeError:
            continue
    return round(auprc, 2)


def compute_auroc(y_true, y_proba):
    thresholds = [0.1 * i for i in range(-1, 12)]
    y_preds = [[int(p > t) for p in y_proba] for t in thresholds]
    recall_scores = [compute_recall(y_true, y_pred) for y_pred in y_preds]
    recall_scores = list(map(lambda x: x if x is not None else 0, recall_scores))
    fpr_scores = [compute_fpr(y_true, y_pred) for y_pred in y_preds]
    fpr_scores = list(map(lambda x: x if x is not None else 0, fpr_scores))
    score_pairs = list(zip(fpr_scores, recall_scores))
    score_pairs = [(0,0)]+list(set(score_pairs)) +[(1,1)]
    score_pairs = sorted(score_pairs, key=lambda x: (x[0], x[1]))
    auroc = 0
    for i in range(1, len(score_pairs)):
        try:
            auroc += (score_pairs[i][0] - score_pairs[i-1][0]) * (score_pairs[i][1] + score_pairs[i-1][1]) / 2
        except TypeError:
            continue
    return round(auroc, 2)


def build_roc_curve(y_true, y_proba, suffix=""):
    thresholds = [0.1 * i for i in range(-1, 12)]
    y_preds = [[int(p > t) for p in y_proba] for t in thresholds]
    recall_scores = [compute_recall(y_true, y_pred) for y_pred in y_preds]
    recall_scores = list(map(lambda x: x if x is not None else 0, recall_scores))
    fpr_scores = [compute_fpr(y_true, y_pred) for y_pred in y_preds]
    fpr_scores = list(map(lambda x: x if x is not None else 0, fpr_scores))
    plt.figure()
    plt.title('Receiver Operating Characteristic Curve')
    plt.plot(fpr_scores, recall_scores, 'b')  # label = 'AUC = %0.2f' % roc_auc
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    app_path = os.path.dirname(os.path.realpath(__name__))
    results_path = app_path + "/results"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    img_path = os.path.join(results_path, "roc_curve" + suffix)
    plt.savefig(img_path)


def build_prc_curve(y_true, y_proba, suffix=""):
    thresholds = [0.1 * i for i in range(-1, 12)]
    y_preds = [[int(p > t) for p in y_proba] for t in thresholds]
    recall_scores = [compute_recall(y_true, y_pred) for y_pred in y_preds]
    recall_scores = list(map(lambda x: x if x is not None else 0, recall_scores))
    precision_scores = [compute_precision(y_true, y_pred) for y_pred in y_preds]
    precision_scores = list(map(lambda x: x if x is not None else 1, precision_scores))
    plt.figure()
    plt.title('Precision Recall Curve')
    plt.plot(recall_scores, precision_scores, 'b')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    app_path = os.path.dirname(os.path.realpath(__name__))
    results_path = app_path + "/results"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    img_path = os.path.join(results_path, "prc_curve" + suffix)
    plt.savefig(img_path)


def split_by_label(data):
    positive = []
    negative = []
    for row in data:
        if row[-1] == 0:
            negative.append(row)
        elif row[-1] == 1:
            positive.append(row)
        else:
            continue
    majority = max([positive, negative], key=len)
    minority = min([positive, negative], key=len)
    return majority, minority


def split_train_test(data, ratio=0.8, stratified=True):
    """
    Split data into random train and test sets based on "ratio". If stratified, preserve proportion of labels.
    """
    def _split_train_test(_data, _ratio=0.8):
        indices = list(range(len(_data)))
        shuffle(indices)
        split_point = int(_ratio*len(indices))
        train_indices = indices[:split_point]
        test_indices = indices[split_point:]
        train = [_data[i] for i in train_indices]
        test = [_data[i] for i in test_indices]
        return train, test
    if stratified:
        majority, minority = split_by_label(data)
        maj_train, maj_test = _split_train_test(majority, ratio)
        min_train, min_test = _split_train_test(minority, ratio)
        return maj_train + min_train, maj_test + min_test
    else:
        _split_train_test(data, ratio)
