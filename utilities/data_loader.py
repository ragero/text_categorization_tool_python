# %% [markdown]
# # Imports

# %%

# Utilities
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder

# %% [markdown]
# # Functions 

def load_csv(path, text_column, class_column, label_encoder=False): 
    df = pd.read_csv(path)
    X = df[text_column].to_numpy()
    y = df[class_column].to_numpy()

    if label_encoder == True: 
        y = np.array(get_label_encoder(y), dtype=np.int)
    else: 
        y =  np.array(y)
    return X, y 

def load_arff(path, sparse=True, class_att= 'class_att', label_encoder=False):
    class_idx = -1
    X = []
    y = []
    class_list = None
    data = False 
    with open(path, 'r') as file: 
        attr_count = 0
        ex_count = 0
        for line in file.readlines(): 
            line = line.lower().strip()
            if data == False: 
                if line.startswith('@attribute'): 
                    if line.find(class_att) >= 0: 
                        class_idx = attr_count  
                        class_list = line[line.find('{')+1:line.find('}')].split(',')
                    attr_count = attr_count + 1
                if line.startswith('@data'): 
                    data = True
            else: 
                example = None 
                class_value = None 
                if sparse == True: 
                    example = line.split(',')
                    if len(example) > 1: 
                        class_value = example[class_idx]
                        example.remove(class_value)
                    else:
                        continue    
                        
                else: 
                    example = np.zeros(attr_count, dtype=np.float32)
                    class_value = ''
                    attrs_values = line[line.find('{')+1:line.find('}')].split(',')
                    if len(attrs_values) > 1: 
                        for attr_value in attrs_values: 
                            parts = attr_value.split(' ')
                            att = int(parts[0])
                            if att == class_idx: 
                                #if not len(parts) > 1: 
                                    #print('Aqui!!!')
                                class_value = parts[1]
                            else: 
                                example[att] = float(parts[1])
                        if class_value == '': 
                            class_value = class_list[0]
                    else: 
                        continue
                X.append(example)
                y.append(class_value)
                ex_count = ex_count + 1
                    
    X = np.array(X, dtype=np.float32)
    if label_encoder == True: 
        y = np.array(get_label_encoder(y), dtype=np.int)
    else: 
        y =  np.array(y)
    
    return X, y

def get_label_encoder(y): 
    labelEncoder = LabelEncoder()
    y = labelEncoder.fit_transform(y)
    return y

loaders = {}
loaders['csv'] = load_csv
loaders['arff'] = load_arff