import sys
from random import seed

from braf.utils.preprocessing import impute, str_column_to_float, load_csv, str_column_to_int
from braf.models.forest import BiasedRandomForestClassifier
from braf.utils.arguments import process_arguments
from braf.utils.evaluation import compute_accuracy, split_train_test, run_cross_validation, generate_output

import logging


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def run_app(argv):

    seed(123)

    # Get params
    params = process_arguments(argv)

    # Load file
    log.info("Processing file.")
    file = params['file']
    data = load_csv(file)

    # Convert values to numerical
    for i in range(len(data[0])-1):
        str_column_to_float(data[1:], i)
    str_column_to_int(data[1:], -1)

    # Impute data set
    log.info("Imputing missing data using KNN imputer (k=3).")
    to_impute = ['SkinThickness', 'BMI', 'Insulin', 'BloodPressure', 'Glucose']
    columns_to_impute = [to_impute.index(c) for c in to_impute]
    impute(data, columns_to_impute, 3)

    #  Run cross validation
    n_folds = 3 if 'n_folds' not in params else params['n_folds']
    log.info("Define {n_trees}-tree BRAF.".format(n_trees=params['n_trees']))
    model = BiasedRandomForestClassifier(**params)
    log.info("Running {n_folds}-fold cross-validation.".format(n_folds=n_folds))
    run_cross_validation(data[1:], model, n_folds)

    # Run on final train-test split and print results
    log.info("Train final model and test using 80-20 split.")
    train_data, test_data = split_train_test(data[1:], stratified=True)
    model.train(train_data)
    actual = [row[-1] for row in test_data]
    probabilities = model.predict_probabilities(test_data)
    predicted = model.predict(test_data)
    dummy = compute_accuracy(actual, [0 for _ in range(len(test_data))])
    print("___\nAccuracy with all 0 predictions: " + str(round(dummy, 2)) + "\n___")
    generate_output(actual, predicted, probabilities, message="Final model results: ", suffix="_final")


if __name__ == '__main__':
    argv = sys.argv
    run_app(argv)
