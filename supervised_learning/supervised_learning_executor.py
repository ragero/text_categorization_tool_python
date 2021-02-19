# %% [markdown]
# # Imports

# %%

# Utilities

import pandas as pd
import numpy as np
import json
import time
from os import path

import sys
sys.path.append(path.join(path.dirname(__file__), '..'))
from utilities.generate_parameters_list import generate_parameters_list
from utilities.data_loader import loaders

#Algorithm
from algorithms.algorithms import dict_algorithms

# Module to carry out the experiments
import supervised_learning as sl

# %%

# %%


# %% [markdown]
# # Test Area

"""config = {
    'path_dataset': '/home/rafael/Área de Trabalho/Projetos/TextCategorizationToolPython/teste/CSTR_sparse.arff',
    'loader': {
        'type': 'arff',
        'sparse': False,
        'class_att': 'class_atr',
        'label_encoder': True
    },
    'path_results': '/home/rafael/Área de Trabalho/Projetos/TextCategorizationToolPython/saida/resultados_teste.csv',
    'validation': {
        'number_folds': 10,
    },
    "algorithms": [
          {
               "name": "LogisticRegression",
               "parameters": {
                    "C": [
                         0.01,
                         0.1,
                         1,
                         10
                    ],
                    "solver": [
                         "saga"
                    ],
                    "n_jobs": [
                         4
                    ]
               }
          }
     ]
   
}"""
# %%
"""with open('config_example_lr_arff_sparse_loader.json','w') as file:
   json.dump(config, file, indent=5)"""

# %%
if __name__ == '__main__': 
    path_json = sys.argv[1]

    with open(path_json, 'r') as file: 
        config = json.load(file)

    X, y = (None, None)
    if 'loader' not in config: 
        raise ValueError('Cofig file must hava a loader entry')
    else: 
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
            sl.execute_exp(X,y,classifier,config)
