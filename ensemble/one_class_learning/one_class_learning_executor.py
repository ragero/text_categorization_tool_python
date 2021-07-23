# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Imports

# %%

import itertools
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from algorithms.DenseAutoencoder import DenseAutoencoder
import one_class_learning as ocl
import pandas as pd
import numpy as np
import json
import time
from os import path

import sys
sys.path.append(path.join(path.dirname(__file__), '..'))
from utilities.data_loader import loaders
from utilities.generate_parameters_list import generate_parameters_list

# %% [markdown]
# # Definitions

# %%
dict_algorithms = {}
dict_algorithms['LocalOutlierFactor'] = LocalOutlierFactor
dict_algorithms['OneClassSVM'] = OneClassSVM
dict_algorithms['EllipticEnvelope'] = EllipticEnvelope
dict_algorithms['IsolationForest'] = IsolationForest
dict_algorithms['DenseAutoencoder'] = DenseAutoencoder

# %% [markdown]
# # Functions


# %%

# %% [markdown]
# # Test Área
#%% 
"""config = {
   "path_dataset": "/media/rafael/DadosCompartilhados/Representacoes/Sequence_of_words_CSV/tr11.mat.csv",
   "loader": {
      "type": "csv",
      "text_column": "Text",
      "class_column": "Class"
   },
   "path_results": "/home/rafael/\u00c1rea de Trabalho/Projetos/TextCategorizationToolPython/saida/resultados_teste.csv",
   "validation": {
      "number_trials": 10,
      "number_labeled_examples": [
         1,
         5,
         10,
         20,
         30
      ],
      "split_type": "random"
   },
   "preprocessing": [
      {
         "method": "TfidfVectorizer",
         "parameters": {
            "min_df": 2
         }
      },
      {
         "method": "NormStandardization"
      }
   ],
   "algorithms": [
      {
         "name": "LocalOutlierFactor",
         "parameters": {
            "n_neighbors": [
               3,
               7,
               11,
            ],
            "metric": [
               "cosine"
            ],
            "novelty": [
               True
            ],
            "n_jobs": [
               4
            ]
         }
      },
      {
         "name": "OneClassSVM",
         "parameters": {
            "nu": [
               0.05,
               0.5,
               0.95
            ],
            "gamma": [
               "scale",
               "auto"
            ],
            "kernel": [
               "linear",
               "rbf"
            ],
            "max_iter": [
               100
            ]
         }
      },
      {
         "name": "IsolationForest",
         "parameters": {
            "n_estimators": [
               10,
               30,
            ],
            "n_jobs": [
               4
            ],
            "random_state": [
               42
            ]
         }
      },
      {
         "name": "EllipticEnvelope",
         "parameters": {
            "assume_centered": [
               True,
               False
            ],
            "contamination": [
               0.1,
               0.2,
            ],
            "random_state": [
               42
            ]
         }
      }
   ]
}"""

# %%


# %%
"""with open('./configs/config_example_ensemble.json','w') as file:
   json.dump(config, file, indent=3)"""


# %%
# path = '/home/rafael/Área de Trabalho/Projetos/TextCategorizationToolPython/one_class_learning/config_example.json'


# %% [markdown]
# # Main

# %%
# Comment the first two lines in case of ruuning the notebook
if __name__ == '__main__':
    path_json = sys.argv[1]

with open(path_json, 'r') as file:
    config = json.load(file)

   print('Loading data...')
   X, y = (None, None)
   loader_type = config['loader']['type']
   loader_params = config['loader']
   del loader_params['type']
   X, y = loaders[loader_type](config['path_dataset'],**loader_params)

   if (X is None) or (y is None):
      raise ValueError('X or y must not be None')


   dict_alg_params = {}
   config_algorithms = config['algorithms']
   for algorithm in config_algorithms:
      parameters = algorithm['parameters']
      parameters_list = generate_parameters_list(parameters)
      for parameter in parameters_list: 
      parameter['name'] = algorithm['name'] 
      dict_alg_params[algorithm['name']] = parameters_list

   lista_algs = []
   for teste in dict_alg_params.keys(): 
   lista_algs.append(dict_alg_params[teste])

   combinations = list(itertools.product(*lista_algs))

   for combinacao in combinations: 
      list_algs = []
      for alg_params in combinacao: 
         current_params = alg_params.copy() 
         algorithm = current_params['name']
         current_params.pop('name')
         parameters = current_params
         classifier = dict_algorithms[algorithm](**parameters)
         list_algs.append(classifier)
      ocl.execute_exp(X, y, list_algs, config)
