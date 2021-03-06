# %% [markdown]
# # Libaries

# Utilities
import time
import pandas as pd
import numpy as np
import json
from os import path
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.append(path.join(path.dirname(__file__), '../../'))
from preprocessing.preprocessing import preprocessors


sys.path.append(path.join(path.dirname(__file__), '../../supervised_learning'))
from supervised_learning import get_evaluation_metrics, check_exp, process_result, get_dataframe

sys.path.append(path.join(path.dirname(__file__), '../../utilities'))
from log_error import log_error

# %% [markdown]
# # Definitions
SEED = 42


# %% [markdown]
# # Functions

# %%
def supervised_learning(X, y, classifier, preprocessor, path_results, num_folds):
    # Splitting data
    np.random.seed(SEED)
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=0)
    # DataFrame to store de results
    current_results = get_dataframe(path_results)

    # Performing k-Fold Cross-Validation
    for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
        # Train test spliting
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Preprocessing pipeline
        if check_exp(current_results, classifier, fold):
            preprocessor.fit(X_train)
            X_train = preprocessor.transform(X_train)
            X_test = preprocessor.transform(X_test)
 
            # Model training
            start_time_classifier_building = time.time()
            classifier.fit(X_train, y_train, preprocessor)
            model_building_time = (time.time() - start_time_classifier_building) / 1000

            # classifier Evaluation
            current_results = process_result(current_results, path_results, classifier, X_test, y_test, fold, model_building_time)


# %%
def execute_exp(X, y, classifier, config):
    path_results = None
    if 'path_results' in config:
        path_results = config['path_results']
    else:
        raise ValueError('Config file must be a "path_result" entry')

    if 'path_dataset' not in config:
        raise ValueError('Config file must be a "path_dataset" entry')
    if 'algorithms' not in config:
        raise ValueError('Config file must be an "algotihms" entry')
    if len(config['algorithms']) == 0:
        raise ValueError('At least one algorithm should be specified')

    num_folds = config['num_folds'] if 'num_folds' in config else 10

    preprocessing = None 
    if 'preprocessing' not in config: 
        preprocessing = preprocessors['PadSequencer']()
    else: 
        parameters_method = preprocessing['parameters'] if 'parameters' in config['preprocessing'] else None
        preprocessing = preprocessors['PadSequencer'](**parameters_method) if parameters_method is not None else preprocessors['PadSequencer']()

    supervised_learning(X, y, classifier, preprocessing, path_results, num_folds)
    """try:
        supervised_learning(X, y, classifier, preprocessing, path_results, num_folds)
    except Exception as Erro:
        log_error('error.log', str(Erro))"""


# %% [markdown]
# # Test Area


# %%
