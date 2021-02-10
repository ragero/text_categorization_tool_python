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

import sys
sys.path.append(path.join(path.dirname(__file__), '..'))
from utilities.generate_parameters_list import generate_parameters_list
from utilities.data_loader import loaders

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
    'path_dataset': '/media/rafael/DadosCompartilhados/Representacoes/Sequence_of_words_CSV/CSTR.csv',
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
            'name': 'DenseAutoencoder',
            'parameters': {
                'layers': [
                    [{'num_neurons':2, 'activation': 'relu'}],
                    [{'num_neurons':6, 'activation': 'relu'}],
                    [{'num_neurons':12, 'activation': 'relu'}],
                    [
                        {'num_neurons':12, 'activation': 'relu'},
                        {'num_neurons':6, 'activation': 'relu'},
                        {'num_neurons':12, 'activation': 'relu'}
                    ]
                ],
                'num_epochs': [200],
                'learning_rate': [0.001]
            }
        },
    ],
    'thresholds' : {
        'fixed': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.85, 0.90, 0.95],
        'six-sigma' : None
    }
    
    
}"""


# %%
"""with open('config_exp.json','w') as file:
   json.dump(config, file, indent=3)"""


# %%
#path = '/home/rafael/Área de Trabalho/Projetos/TextCategorizationToolPython/one_class_learning/config_example.json'


# %% [markdown]
# # Main

# %%
# Comment the first two lines in case of ruuning the notebook
if __name__ == '__main__': 
    path_json = sys.argv[1]

    with open(path_json, 'r') as file: 
        config = json.load(file)

    X, y = loaders['csv'](config['path_dataset'],'text','class')

    config_algorithms = config['algorithms']
    for algorithm in config_algorithms: 
        parameters = algorithm['parameters']
        parameters_list = generate_parameters_list(parameters)
        for parameters in parameters_list: 
            print(parameters)
            classifier = dict_algorithms[algorithm['name']](**parameters)
            ocl.execute_exp(X,y,classifier,config)
        




# %%
