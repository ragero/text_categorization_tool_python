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
"""config = {
    'path_dataset': '/home/rafael/Downloads/iris.csv',
    'path_results': '/home/rafael/Área de Trabalho/Projetos/TextCategorizationToolPython/saida/resultados_teste.csv',
    'validation': {
        'number_trials': 10,
        'number_labeled_examples': [1, 5, 10, 20, 30],
        'split_type': 'random',
    },
    'preprocessing': [
        {
            'method': 'TfidfVectorizer',
            'parameters': {
                'min_df' : 2
            }
        },
        {
            'method': 'SumStandardization'
        }
    ],
    'algorithms': [
        {
            'name': 'LocalOutlierFactor',
            'parameters': {
                'n_neighbors' : [1, 5, 9, 13, 17, 21],
                'metric': ['cosine','euclidean'],
                'novelty': [True],
                'n_jobs': [4]
            }
        },
        {
            'name': 'OneClassSVM',
            'parameters': {
                'nu': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,  0.85, 0.9, 0.95],
                'gamma': ['scale','auto'],
                'kernel': ['linear', 'rbf'],
                'max_iter': [100]
            }
        },
        {
            'name': 'EllipticEnvelope',
            'parameters': {
                'assume_centered': [True, False],
                'contamination': [0.1, 0.2, 0.3, 0.4],
                'random_state' : [42]
            }
        },
        {
            'name': 'IsolationForest',
            'parameters': {
                'n_estimators' : [10,30,50,70,90],
                'n_jobs': [4],
                'random_state' : [42]
            }
        },
        {
            'name': 'DenseAutoencoder',
            'parameters': {
                'encoding_dim': [2],
                'num_epochs': [200],
                'learning_rate': [0.01]
            }
        },
    ],
    'thresholds' :
        {'fixed': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.85, 0.90, 0.95],
        'six-sigma' : None}
}"""


# %%
"""config = {
     "path_dataset": "/media/rafael/DadosCompartilhados/Representacoes/FakeNews/baseFacom_filter_nonsparsetosparse.arff",
     "loader": {
          "type": "arff",
          "sparse": False,
          "class_att": "class_atr"
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
               "method": "NormStandardization"
          }
     ],
     "algorithms": [
          {
               "name": "DenseAutoencoder",
               "parameters": {
                    "layers": [
                         
                            {
                                "hidden": [
                                   {
                                      "type": "dense",
                                      "units": 2,
                                      "activation": "relu"
                                   },
                                   {
                                      "type": "dropout",
                                      "rate": 0.5
                                   }
                                ],
                                "output": {
                                   "activation": "sigmoid"
                                }
                             }
                         ,
                         
                            {
                                "hidden": [
                                   {
                                      "type": "dense",
                                      "units": 6,
                                      "activation": "relu"
                                   },
                                   {
                                      "type": "dropout",
                                      "rate": 0.5
                                   }
                                ],
                                "output": {
                                   "activation": "sigmoid"
                                }
                             }
                         ,
                         
                            {
                                "hidden": [
                                   {
                                      "type": "dense",
                                      "units": 12,
                                      "activation": "relu"
                                   },
                                   {
                                      "type": "dropout",
                                      "rate": 0.5
                                   }
                                ],
                                "output": {
                                   "activation": "sigmoid"
                                }
                             }
                         
                    ],
                    "num_epochs": [
                         200
                    ],
                    "learning_rate": [
                         0.01
                    ],
                    "loss": ["binary_crossentropy"]
               }
          }
     ],
     "thresholds": {
          "fixed": [
               0.05,
               0.1,
               0.15,
               0.2,
               0.25,
               0.3,
               0.35,
               0.4,
               0.45,
               0.5,
               0.55,
               0.6,
               0.65,
               0.7,
               0.75,
               0.85,
               0.9,
               0.95
          ],
          "six-sigma": None
     }
}"""


# %%
"""with open('./configs/config_example_linear_dense_autoencoder_arff_sparse.json','w') as file:
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

    X, y = (None, None)
    loader_type = config['loader']['type']
    loader_params = config['loader']
    del loader_params['type']
    X, y = loaders[loader_type](config['path_dataset'],**loader_params)

    if (X is None) or (y is None):
        raise ValueError('X or y must not be None')

    config_algorithms = config['algorithms']
    for algorithm in config_algorithms:
        parameters = algorithm['parameters']
        parameters_list = generate_parameters_list(parameters)
        for parameters in parameters_list:
            print(parameters)
            classifier = dict_algorithms[algorithm['name']](**parameters)
            ocl.execute_exp(X, y, classifier, config)
