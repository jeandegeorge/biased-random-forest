import logging


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def process_arguments(argv):
    try:
        params = {
            "file": str(argv[argv.index('--file') + 1]),
            "p": float(argv[argv.index('--p') + 1]),
            "k": int(argv[argv.index('--k') + 1])
        }
        if '--n_folds' in argv:
            params["n_folds"] = int(argv[argv.index('--n_folds') + 1])
        if '--max_depth' in argv:
            params["max_depth"] = int(argv[argv.index('--max_depth') + 1])
        if '--min_size' in argv:
            params["min_size"] = int(argv[argv.index('--min_size') + 1])
        if '--sample_size' in argv:
            params["sample_size"] = int(argv[argv.index('--sample_size') + 1])
        if '--n_trees' in argv:
            params["n_trees"] = int(argv[argv.index('--n_trees') + 1])
        if '--n_features' in argv:
            params["n_features"] = int(argv[argv.index('--n_features') + 1])
        return params
    except (ValueError, TypeError):
        log.error("Some arguments were not defined correctly.")

