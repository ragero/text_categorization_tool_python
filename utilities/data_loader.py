# %% [markdown]
# # Imports

# %%

# Utilities
import pandas as pd 

# %% [markdown]
# # Functions 

def load_csv(path, text_column, class_column): 
    df = pd.read_csv(path)
    X = df[text_column].to_numpy()
    y = df[class_column].to_numpy()
    return X, y 

loaders = {}
loaders['csv'] = load_csv