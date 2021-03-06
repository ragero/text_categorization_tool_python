# %% [markdown]
# # Libaries

# Utilities
import time
import pandas as pd
import numpy as np
import json
from os import path

import sys
sys.path.append(path.join(path.dirname(__file__), '..'))
from preprocessing.preprocessing import preprocessors
from utilities.log_error import log_error



# Evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold


# %% [markdown]
# # Definitions
SEED = 42


# %% [markdown]
# # Functions

# %%
def get_evaluation_metrics(classifier, X_test, y_test, iteration, model_building_time):

  evaluation = {}
  start_time_classification = time.time()
  predictions = classifier.predict(X_test)
  elapsed_time_classification = (
      time.time() - start_time_classification) / 1000

  evaluation['Algorithm'] = classifier.__class__.__name__
  evaluation['Parameters'] = str(classifier.get_params())
  evaluation['Iteration'] = iteration

  evaluation['Accuracy'] = accuracy_score(y_test, predictions)

  evaluation['Macro_Precision'] = precision_score(y_test, predictions, average='macro', zero_division=0)
  evaluation['Macro_Recall'] = recall_score(y_test, predictions, average='macro', zero_division=0)
  evaluation['Macro_F1'] = f1_score(y_test, predictions, average='macro')
  #evaluation['Macro_AUC_ROC'] = roc_auc_score(y_test, predictions, average='macro')

  evaluation['Micro_Precision'] = precision_score(y_test, predictions, average='micro', zero_division=0)
  evaluation['Micro_Recall'] = recall_score(y_test, predictions, average='micro', zero_division=0)
  evaluation['Micro_F1'] = f1_score(y_test, predictions, average='micro')
  #evaluation['Micro_AUC_ROC'] = roc_auc_score(y_test, predictions, average='micro')

  evaluation['Confusion_Matrix'] = confusion_matrix(y_test, predictions).tolist()

  evaluation['Building_Time'] = model_building_time
  evaluation['Classification_Time'] = elapsed_time_classification

  evaluation['Memory'] = sys.getsizeof(classifier) / 1024

  return evaluation

# %%
def check_exp(results, classifier, iteration):
    if len(results[(results['Algorithm'] == classifier.__class__.__name__) & (results['Parameters'] == str(classifier.get_params())) & (results['Iteration'] == iteration)]) > 0:
        return False
    else:
        return True

# %%
def process_result(current_results, path_results, classifier, X_test, y_test, iteration,model_building_time):
    
    result = get_evaluation_metrics(classifier, X_test, y_test, iteration, model_building_time)
    print(result, '\n')
    current_results = current_results.append(result, ignore_index=True)
    current_results.to_csv(path_results, index=False)
    return current_results

# %%
def supervised_learning(X, y, classifier, preprocessing_pipeline, path_results, num_folds):
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

    #Preprocessing pipeline
    if check_exp(current_results, classifier, fold):
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

      # Model training
      start_time_classifier_building = time.time()
      classifier.fit(X_train, y_train)
      model_building_time = (time.time() - start_time_classifier_building) / 1000

      # classifier Evaluation
      current_results = process_result(current_results, path_results, classifier, X_test, y_test, fold,model_building_time)


# %%


def get_dataframe(path_results):
  results = None
  if (path.exists(path_results)):
    results = pd.read_csv(path_results)
  else:
    results = pd.DataFrame(columns=[
      'Algorithm',
      'Parameters',
      'Iteration',
      'Accuracy',
      'Macro_Precision',
      'Macro_Recall',
      'Macro_F1',
      'Micro_Precision',
      'Micro_Recall',
      'Micro_F1',
      'Confusion_Matrix',
      'Memory',
      'Building_Time',
      'Classification_Time'
    ])

  return results

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

  preprocessing_pipeline = []
  if 'preprocessing' in config:
     for preprocessing in config['preprocessing']:
            preprocessing_method = preprocessors[preprocessing['method']]
            parameters_method = preprocessing['parameters'] if 'parameters' in preprocessing else None
            preprocessing_pipeline.append(preprocessing_method(**parameters_method) if parameters_method != None else preprocessing_method())

  #supervised_learning(X, y, classifier, preprocessing_pipeline, path_results, num_folds)
  try:
      supervised_learning(X, y, classifier, preprocessing_pipeline, path_results, num_folds)
  except Exception as Erro:
      log_error('error.log', str(Erro))

    

# %% [markdown]
# # Test Area










# %%
