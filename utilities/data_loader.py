# %% [markdown]
# # Imports

# %%

# Utilities
import pandas as pd 

# %% [markdown]
# # Functions 

def load_csv(path, text_column, class_column): 
    df = pd.read_csv(path)
    X = df[text_column]
    y = df[class_column]
    return X, y 

loaders = {}
loaders['csv'] = load_csv