# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Libaries

# %%
# Data Structures and Utilities
import numpy as np
import pandas as pd
import time
import os
import sys
import sys
# sys.path.append('..')
#from utilities.log_errors import log_error


# Learning evaluation
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

# Preprocessing Algorithms
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing.standardizations import NormStandardization
from preprocessing.standardizations import SumStandardization

# %% [markdown]
# # Definitions

# %%
SEED = 42

dict_preprocessing_methods = {}
dict_preprocessing_methods['TfidfVectorizer'] = TfidfVectorizer
dict_preprocessing_methods['NormStandardization'] = NormStandardization
dict_preprocessing_methods['SumStandardization'] = SumStandardization


# %% [markdown]
# # Functions

# %%
# Split the dataset into train and test data

# split_type: type os splitting of the examples from the interest class: "cross-validation" or "random"
# number: number of folds in case of type == "cross-validation", or number or examples in case of type == "random"
def get_indexes(data, split_type, number_trials, number_examples):
    indexes = []
    np.random.seed(SEED)  # seed for random selections
    if split_type == 'cross-validation':
        kf = KFold(n_splits=number_trials, shuffle=True, random_state=SEED)
        for ids_train, ids_test in kf.split(data):
            indexes_train = data[ids_train]
            indexes.append(indexes_train)
    elif split_type == 'random':
        for it in range(number_trials):
            indexes.append(np.random.choice(data, size=min(
                number_examples, len(data)), replace=False))
    else:
        raise ValueError(
            'Unsuported split type. Please, use split_type = {"cross-validation","random"}.')
    return indexes


# %%
def get_train_test_data(X, all_indexes, indexes_train):
    indexes_test = list(set(all_indexes) - set(indexes_train))
    return X[indexes_train], X[indexes_test]


# %%
def get_classes_test(y, classe, all_indexes, indexes_train):
    indexes_test = list(set(all_indexes) - set(indexes_train))
    y_test = np.ones(len(indexes_test), dtype=np.int)
    for i, element in enumerate(y[indexes_test]):
        if element != classe:
            y_test[i] = -1
    return y_test


# %%
def get_evaluation_metrics(classifier, X_test, y_test, threshold_type, threshold_value, classe, num_labeled_exs, it_number, model_building_time=0):

    evaluation = {}
    start_time_classification = time.time()
    predictions = classifier.predict(X_test)
    scores = classifier.decision_function(X_test)

    print('Predictions:', predictions)
    print('Scores:', scores)
    elapsed_time_classification = (
        time.time() - start_time_classification) / 1000

    evaluation['Algorithm'] = classifier.__class__.__name__
    evaluation['Parameters'] = str(classifier.get_params())
    evaluation['Class'] = classe
    evaluation['Threshold_Type'] = threshold_type
    evaluation['Threshold_Value'] = threshold_value
    evaluation['Number_Labeled_Examples'] = num_labeled_exs
    evaluation['Iteration'] = it_number
    evaluation['Accuracy'] = accuracy_score(y_test, predictions)
    evaluation['Precision'] = precision_score(y_test, predictions)
    evaluation['Recall'] = recall_score(y_test, predictions)
    evaluation['F1'] = f1_score(y_test, predictions)
    evaluation['ROC_AUC'] = roc_auc_score(y_test, scores, average=None)
    evaluation['Confusion_Matrix'] = confusion_matrix(y_test, predictions).tolist()
    evaluation['Building_Time'] = model_building_time
    evaluation['Classification_Time'] = elapsed_time_classification
    evaluation['Memory'] = sys.getsizeof(classifier) / 1024

    return evaluation


# %%
def get_data_frame(path_results):
    results = None
    if (os.path.exists(path_results)):
        results = pd.read_csv(path_results)
    else:
        results = pd.DataFrame(columns=['Algorithm',
                                        'Parameters',
                                        'Threshold_Type',
                                        'Threshold_Value',
                                        'Class',
                                        'Number_Labeled_Examples',
                                        'Iteration',
                                        'Accuracy',
                                        'Precision',
                                        'Recall',
                                        'F1',
                                        'ROC_AUC',
                                        'Confusion_Matrix',
                                        'Building_Time',
                                        'Classification_Time',
                                        'Memory'
                                        ])
    return results


def compute_sig_sigma_thresholds(classifier, X_train):
    thresholds = []
    scores = classifier.decision_function(X_train)
    mean = scores.mean()
    std = scores.std()
    for i in range(-3, 4):
        threshold = mean + i*std
        threshold = min(1, threshold)
        threshold = max(0, threshold)
        thresholds.append(threshold)
    return thresholds

# %%


def process_result(current_results, path_results, classifier, X_test, y_test, threshold_type, threshold_value, classe, num_labeled_exes, it, model_building_time):
    result = get_evaluation_metrics(classifier, X_test, y_test, threshold_type, threshold_value, classe, num_labeled_exes, it, model_building_time)
    print(result, '\n')
    current_results = current_results.append(result, ignore_index=True)
    current_results.to_csv(path_results, index=False)
    return current_results

# %%


def check_exp(results, classifier, classe, iteration, num_labeled_exes):
    if len(results[(results['Algorithm'] == classifier.__class__.__name__) & (results['Parameters'] == str(classifier.get_params())) & (results['Class'] == classe) & (results['Number_Labeled_Examples'] == num_labeled_exes) & (results['Iteration'] == iteration)]) > 0:
        return False
    else:
        return True


# %%
#X: dada
#y: classes
# split_type: type os splitting of the examples from the interest class: "cross-validation" or "random"
# classifier: OCL algorithm
# number_trials: number of folds in case of split_type == "cross-validation", or number or repetitions in case of split_type == "random"
# number_examples: number of labeled_examples if split_type == "random"
def one_class_learning(X, y, classifier, thresholds, preprocessing_pipeline=[], path_results='results.csv', split_type='cross-validation', number_trials=10, number_examples=10):
    current_results = get_data_frame(path_results)
    all_indexes = set(range(len(X)))
    classes = np.unique(y)
    for classe in classes:
        classe_indexes = np.argwhere(y == classe).reshape(-1)
        for it, indexes_train in enumerate(get_indexes(classe_indexes, split_type, number_trials, number_examples)):
            X_train, X_test = get_train_test_data(X, all_indexes, indexes_train)
            y_test = get_classes_test(y, classe, all_indexes, indexes_train)
            if len(np.unique(y_test)) < 2 or len(y_test) == 0: 
                continue
            num_labeled_exes = len(X_train)
            if check_exp(current_results, classifier, classe, it, num_labeled_exes):
                if preprocessing_pipeline != None:
                    for preprocessor in preprocessing_pipeline:
                        preprocessor.fit(X_train)
                        X_train = preprocessor.transform(X_train)
                        try:
                            X_train = X_train.todense()
                        except AttributeError:
                            pass
                        X_test = preprocessor.transform(X_test)
                        try:
                            X_test = X_test.todense()
                        except AttributeError:
                            pass
                
                
                start_time_building = time.time()
                classifier.fit(X_train)
                elapsed_time_building = (
                time.time() - start_time_building) / 1000
                if thresholds != None:
                    if 'fixed' in thresholds:
                        for threshold in thresholds['fixed']:
                            classifier.set_threshold(threshold)
                            current_results = process_result(
                                current_results, path_results, classifier, X_test, y_test, 'Fixed', threshold, classe, num_labeled_exes, it, elapsed_time_building)
                    if 'six-sigma' in thresholds:
                        six_sigma_thresholds = compute_sig_sigma_thresholds(classifier, X_train)
                        for threshold in six_sigma_thresholds:
                            classifier.set_threshold(threshold)
                            current_results = process_result(current_results, path_results, classifier, X_test, y_test, '6-sigma', threshold, classe, num_labeled_exes, it, elapsed_time_building)
                else:
                    current_results = process_result(current_results, path_results, classifier, X_test, y_test, 'None', 'None', classe, num_labeled_exes, it, elapsed_time_building)


'preprocessing'
# %%
# # Testing
#preprocessing_pipeline = [TfidfVectorizer(min_df=2), NormStandardization()]


# %%
def execute_exp(X, y, classifier, config):
    path_results = None
    if 'path_results' not in config:
        raise ValueError('Config file must be a "path_result" entry')
    else:
        path_results = config['path_results']
    if 'path_dataset' not in config:
        raise ValueError('Config file must be a "path_dataset" entry')
    if 'algorithms' not in config:
        raise ValueError('Config file must be a "algorithm" entry')
    if len(config['algorithms']) == 0:
        raise ValueError('At least one algorhtm should be specified')

    number_trials = None
    if 'number_trials' not in config:
        number_trials = 10
    else:
        number_trials = config['validation']

    split_type = None
    if 'split_type' not in config['validation']:
        split_type = 'cross-validation'
    else:
        split_type = config['validation']['split_type']

    number_labeled_examples = []
    if ('number_labeled_examples' not in config['validation']) or split_type == 'cross-validation':
        number_labeled_examples = [None]
    else:
        number_labeled_examples = config['validation']['number_labeled_examples']

    thresholds = None
    if 'thresholds' in config:
        thresholds = config['thresholds']

    preprocessing_pipeline = []
    if 'preprocessing' in config:
        for preprocessing in config['preprocessing']:
            preprocessing_method = dict_preprocessing_methods[preprocessing['method']]
            parameters_method = preprocessing['parameters'] if 'parameters' in preprocessing else None
            preprocessing_pipeline.append(preprocessing_method(
                parameters_method) if parameters_method != None else preprocessing_method())

    for nle in number_labeled_examples:
        one_class_learning(X, y, classifier, thresholds, preprocessing_pipeline, path_results,
                           split_type=split_type, number_trials=number_trials, number_examples=nle)
        """try:
            one_class_learning(X, y, classifier, thresholds preprocessing_pipeline, path_results,
                               split_type=split_type, number_trials=number_trials, number_examples=nle)
        except Exception as Erro:
            log_error(str(Erro))"""

    print('Done')


# %%
