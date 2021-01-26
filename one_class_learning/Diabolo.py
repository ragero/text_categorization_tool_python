# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Imports

# %%
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from scipy.spatial.distance import cosine

# %% [markdown]
# # Functions

# %%
def sim_cosine(vec1, vec2): 
  return 1 - cosine(vec1, vec2)


# %%
def standardization(X):
    for i in range(len(X)):
        X[i] = X[i] / X[i].max()
    return X

# %% [markdown]
# # Diabolo Class

# %%
class Diabolo(object):

    def __init__(self, input_size, threshold): 
        self.input_size = input_size
        self.threshold = threshold
        encoding_dim = 2
        input = tf.keras.Input(shape=(input_size,))
        encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input)
        decoded = tf.keras.layers.Dense(input_size, activation='sigmoid')(encoded)
        autoencoder = tf.keras.Model(input, decoded) 
        autoencoder.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy')

        self.model = autoencoder
    
    def get_params(self):
        return{'threshold': self.threshold}

    def fit(self,X,epochs=500, shuffle=True, batch_size=1):
        self.model.fit(X,X, epochs=epochs, shuffle=shuffle, batch_size=batch_size) 

    def decision_function(self,X): 
        scores = np.zeros(len(X), dtype=np.float32)
        model_outputs = self.model.predict(X)
        for i, output in enumerate(model_outputs): 
            scores[i] = sim_cosine(X[i], output)
        return scores 

    def predict(self,X): 
        predictions = np.zeros(len(X))
        scores = self.decision_function(X)
        for i,score in enumerate(scores): 
            predictions[i] = 1 if score > self.threshold else 0
        return predictions


# %%


# %% [markdown]
# # Test area

# %%



