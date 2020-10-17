#####INSTRUCTIONS TO RUN:

$cd [app path]
$python3 setup.py build install pytest sdist
$pip install dist/braf-0.0.1.tar.gz
$braf.sh --file [file directory]/diabetes.csv --p 0.5 --k 10 --max_depth 10 --min_size 1 --sample_size 1 --n_trees 5 --n_features 5

The curves will be saved at the same level  as the directory where you run the command from in a folder called results.

Only files formatted just like diabetes.csv can be fed to the app (e.g. missing values must be 0 not None type).

#####PARAMETERS:

- *p* is the proportion of trees trained on the critical data set (float, 0<=p<=1).
- *k* is the number of closest neighbors used to generate critical data set (int, k>=0).
- *max_depth* is the maximum depth of the trees (int, max_depth>0).
- *min_size* is the maximum size of a data set resulting from a tree split (int, min_size>0).
- *sample_size* is the size of the bagging sub-samples used to train the RF classifier (int, sample_size>0).
- *n_trees* is the number of trees per RF classifier (int, n_trees>0).
- *n_features* is the number of features the tree is built on, picked at random (int, [number of independent vars]>=n_features>0).
- *n_folds* is the number of cross validation folds (int, n_folds>0).

#####NOTES:

This code reproduces the logic described in "Biased Random Forest For Dealing With the Class Imbalance Problem" by Bader-El-Den, Teitei and Perry.
The goal is to tweak the standard RF algorithm to better deal with class imbalance.
