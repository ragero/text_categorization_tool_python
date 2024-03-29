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

# %%
config = {
   "path_dataset": "/media/rafael/DadosCompartilhados/Representacoes/Sequence_of_words_CSV/CSTR.csv",
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
      "split_type": "cross-validation"
   },
   "preprocessing": [
      {
         "method": "TfidfVectorizer",
         "parameters": {
            "min_df": 2
         }
      }
   ],
   "algorithms": [
      {
         "name": "OneClassSVM",
         "parameters": {
            "nu": [
               0.50,
               0.95
            ],
            "gamma": [
               "scale",
            ],
            "kernel": [
               "linear",
            ],
            "max_iter": [
               100
            ]
         }
      }, 
      {
         "name": "LocalOutlierFactor",
         "parameters": {
            "n_neighbors": [
               1,
               5
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
      }
   ]
}

# %%
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
               1,
               5,
               9,
               13,
               17,
               21
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
      }
   ]
}"""






# %%
"""with open('./configs/config_example_linear_dense_autoencoder_arff_sparse_mariana.json','w') as file:
   json.dump(config, file, indent=3)"""


# %%
# path = '/home/rafael/Área de Trabalho/Projetos/TextCategorizationToolPython/one_class_learning/config_example.json'


# %% [markdown]
# # Main

# %%
# Comment the first two lines in case of ruuning the notebook
"""if __name__ == '__main__':
   path_json = sys.argv[1]

   with open(path_json, 'r') as file:
      config = json.load(file)"""

X, y = (None, None)
loader_type = config['loader']['type']
loader_params = config['loader']
del loader_params['type']
X, y = loaders[loader_type](config['path_dataset'],**loader_params)

if (X is None) or (y is None):
   raise ValueError('X or y must not be None')

config_algorithms = config['algorithms']

dict_alg_params = {}

config_algorithms = config['algorithms']
for algorithm in config_algorithms:
   parameters = algorithm['parameters']
   parameters_list = generate_parameters_list(parameters)
   for parameter in parameters_list: 
     parameter['name'] = algorithm['name'] 
   dict_alg_params[algorithm['name']] = parameters_list



print('teste')
for parameters in parameters_list:
   print(parameters)
   classifier = dict_algorithms[algorithm['name']](**parameters)
   ocl.execute_exp(X, y, classifier, config)
