# %% [markdown]
# # Imports

# %%

# Utilities

import supervised_deep_learning as sdl
from algorithms.algorithms import dict_algorithms
from utilities.data_loader import loaders
from utilities.generate_parameters_list import generate_parameters_list
import pandas as pd
import numpy as np
import json
import time
from os import path

import sys
sys.path.append(path.join(path.dirname(__file__), '../..'))

# Algorithm

# Module to carry out the experiments

# %%

# %%


# %% [markdown]
# # Test Area

config = {
    'path_dataset': '/media/rafael/DadosCompartilhados/Datasets/Textos em CSVs/CSTR.csv',
    'loader': {
        'type': 'csv',
        'text_column': 'text_preprocessed',
        'class_column': 'class',
        'label_encoder': True
    },
    'path_results': '/home/rafael/Área de Trabalho/Projetos/TextCategorizationToolPython/saida/resultados_teste.csv',
    'validation': {
        'number_folds': 10,
    },
    "algorithms": [
        {
            "name": "MyDeepLearning",
            "parameters": {
                'epochs': [100],
                'batch_size': [32],
                'learning_rate': [0.001],
                "layers": [
                    {   
                        "embedding": {
                            'output_dim': 50,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 50,
                                "kernel_size": 2,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },
                    {   
                        "embedding": {
                            'output_dim': 50,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 50,
                                "kernel_size": 3,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },
                    {   
                        "embedding": {
                            'output_dim': 50,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 50,
                                "kernel_size": 4,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },
                    {   
                        "embedding": {
                            'output_dim': 50,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 50,
                                "kernel_size": 5,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },
                    
                    
                    
                    {   
                        "embedding": {
                            'output_dim': 50,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 100,
                                "kernel_size": 2,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },
                    {   
                        "embedding": {
                            'output_dim': 50,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 100,
                                "kernel_size": 3,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },
                    {   
                        "embedding": {
                            'output_dim': 50,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 100,
                                "kernel_size": 4,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },
                    {   
                        "embedding": {
                            'output_dim': 50,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 100,
                                "kernel_size": 5,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },


                    {   
                        "embedding": {
                            'output_dim': 100,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 100,
                                "kernel_size": 2,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },
                    {   
                        "embedding": {
                            'output_dim': 50,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 100,
                                "kernel_size": 3,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },

                    {   
                        "embedding": {
                            'output_dim': 50,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 100,
                                "kernel_size": 4,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },
                    {   
                        "embedding": {
                            'output_dim': 50,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 100,
                                "kernel_size": 5,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },

                    
                    {   
                        "embedding": {
                            'output_dim': 100,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 50,
                                "kernel_size": 2,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },
                    {   
                        "embedding": {
                            'output_dim': 100,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 50,
                                "kernel_size": 3,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },
                    {   
                        "embedding": {
                            'output_dim': 100,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 50,
                                "kernel_size": 4,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },
                    {   
                        "embedding": {
                            'output_dim': 100,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 50,
                                "kernel_size": 5,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },
                    
                    
                    
                    {   
                        "embedding": {
                            'output_dim': 100,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 100,
                                "kernel_size": 2,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },
                    {   
                        "embedding": {
                            'output_dim': 100,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 100,
                                "kernel_size": 3,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },
                    {   
                        "embedding": {
                            'output_dim': 100,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 100,
                                "kernel_size": 4,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },
                    {   
                        "embedding": {
                            'output_dim': 100,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 100,
                                "kernel_size": 5,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },


                    {   
                        "embedding": {
                            'output_dim': 100,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 100,
                                "kernel_size": 2,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },
                    {   
                        "embedding": {
                            'output_dim': 100,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 100,
                                "kernel_size": 3,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },

                    {   
                        "embedding": {
                            'output_dim': 100,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 100,
                                "kernel_size": 4,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },
                    {   
                        "embedding": {
                            'output_dim': 100,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 100,
                                "kernel_size": 5,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },


                    
                    {   
                        "embedding": {
                            'output_dim': 300,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 50,
                                "kernel_size": 2,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },
                    {   
                        "embedding": {
                            'output_dim': 300,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 50,
                                "kernel_size": 3,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },
                    {   
                        "embedding": {
                            'output_dim': 300,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 50,
                                "kernel_size": 4,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },
                    {   
                        "embedding": {
                            'output_dim': 300,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 50,
                                "kernel_size": 5,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },
                    
                    
                    
                    {   
                        "embedding": {
                            'output_dim': 300,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 100,
                                "kernel_size": 2,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },
                    {   
                        "embedding": {
                            'output_dim': 300,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 100,
                                "kernel_size": 3,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },
                    {   
                        "embedding": {
                            'output_dim': 300,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 100,
                                "kernel_size": 4,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },
                    {   
                        "embedding": {
                            'output_dim': 300,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 100,
                                "kernel_size": 5,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },


                    {   
                        "embedding": {
                            'output_dim': 300,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 100,
                                "kernel_size": 2,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },
                    {   
                        "embedding": {
                            'output_dim': 300,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 100,
                                "kernel_size": 3,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },

                    {   
                        "embedding": {
                            'output_dim': 300,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 100,
                                "kernel_size": 4,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },
                    {   
                        "embedding": {
                            'output_dim': 300,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 100,
                                "kernel_size": 5,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },

                    {   
                        "embedding": {
                            'output_dim': 300,
                            'trainable': True
                        },
                        "hidden": [
                            {
                                "type": "Conv1D",
                                "filters": 100,
                                "kernel_size": 2,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 3
                            },
                            {
                                "type": "Conv1D",
                                "filters": 100,
                                "kernel_size": 5,
                                "activation":'relu'
                            },
                            {
                                "type": "MaxPooling1D",
                                "pool_size": 2
                            },
                            {
                                "type": "Flatten",
                            },

                        ],
                        "output": {
                            "activation": "softmax",
                            "regularizer": 'l2'
                        }
                    },

                ]
            }
        }
    ]

}
# %%
with open('./configs/config_CNN.json', 'w') as file:
    json.dump(config, file, indent=5)

# %%
"""if __name__ == '__main__': 
    path_json = sys.argv[1]

    with open(path_json, 'r') as file: 
        config = json.load(file)"""

X, y = (None, None)
if 'loader' not in config:
    raise ValueError('Cofig file must hava a loader entry')
else:
    loader_type = config['loader']['type']
    loader_params = config['loader']
    del loader_params['type']
    X, y = loaders[loader_type](config['path_dataset'], **loader_params)

if (X is None) or (y is None):
    raise ValueError('X or y must not be None')


config_algorithms = config['algorithms']
for algorithm in config_algorithms:
    parameters = algorithm['parameters']
    parameters_list = generate_parameters_list(parameters)
    for parameters in parameters_list:
        print(parameters)
        classifier = dict_algorithms[algorithm['name']](**parameters)
        sdl.execute_exp(X, y, classifier, config)
print('Done!')
