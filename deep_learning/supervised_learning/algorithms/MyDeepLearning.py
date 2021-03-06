# %% [markdown]
# # Libaries
import numpy as np
import tensorflow as tf 

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# from tensorflow.keras.layers import Dense, Embedding, Input
# from tensorflow.keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
# from tensorflow.keras.models import Model 
# from tensorflow.keras.optimizers import Adam

layers = {}
layers['LSTM'] = tf.keras.layers.LSTM
layers['lstm'] = tf.keras.layers.LSTM 
layers['GRU'] = tf.keras.layers.GRU
layers['gru'] = tf.keras.layers.GRU
layers['Dropout'] = tf.keras.layers.Dropout
layers['dropout'] = tf.keras.layers.Dropout
layers['Dense'] = tf.keras.layers.Dense
layers['dense'] = tf.keras.layers.Dense
layers['GlobalMaxPool1D'] = tf.keras.layers.GlobalMaxPool1D
layers['globalmaxpool1D'] = tf.keras.layers.GlobalMaxPool1D
layers['Embedding'] = tf.keras.layers.Embedding
layers['embedding'] = tf.keras.layers.Embedding
layers['GlobalAveragePooling1D'] = tf.keras.layers.GlobalAveragePooling1D
layers['globalaveragepooling1d'] = tf.keras.layers.GlobalAveragePooling1D
layers['Conv1D'] = tf.keras.layers.Conv1D 
layers['conv1d'] = tf.keras.layers.Conv1D
layers['MaxPooling1D'] = tf.keras.layers.MaxPooling1D
layers['maxpooling1D'] = tf.keras.layers.MaxPooling1D



# %% [markdown]
# # Class
class MyDeepLearning(object): 

    def __init__(self, epochs = 10, batch_size = 32, learning_rate = 0.001, layers=None): 
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.layers = layers
        self.model = None 

    def get_params(self):
        return {
            'layers': self.layers,
            'batch_size': self.batch_size,
            'num_epochs': self.epochs,
            'learning_rate': self.learning_rate,
        }


    def fit(self, X, y, preprocessor): 
        
        # Creating the embedding layer
        embedding_layer = layers['embedding'](input_dim=preprocessor.num_words,input_length=preprocessor.max_sequence_lenght, **self.layers['embedding'])
        
        # Input Layer
        input_ = tf.keras.layers.Input(shape=(preprocessor.max_sequence_lenght,))
        
        # Connecting input layer with hidden layer
        hidden = embedding_layer(input_)
        
        # Creating other hidden layers
        for i,layer in enumerate(self.layers['hidden']):
            current_layer = layer.copy()
            layer_type = current_layer['type']
            layer_params = current_layer
            del layer_params['type']
            hidden =  layers[layer_type](**layer_params)(hidden)

        # Creating the output layer
        output = layers['dense'](len(np.unique(y)), activation=self.layers['output']['activation'])(hidden)

      
        # Creting and compiling the model 
        self.model = tf.keras.models.Model(input_, output)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(self.learning_rate), metrics=['accuracy'])

        # Training the model
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size)

    def predict(self, X): 
        predictions = self.model.predict(X)
        output = np.zeros(len(X))
        for i,prediction in enumerate(predictions):
            output[i] = np.argmax(prediction)
        return output

   