from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from statistics import mean

# %% [markdown]
# # Libaries


# %% [markdown]
# # Class

class PadSequencer(object): 

    def __init__(self, max_vocab_size = 20000, max_sequence_lenght='average'): 
        self.max_sequence_lenght = max_sequence_lenght
        self.max_vocab_size = max_vocab_size
        self.tokenizer = None 
        self.num_words = 0

    def fit(self, X): 
        
        self.tokenizer = Tokenizer(num_words=self.max_vocab_size)
        self.tokenizer.fit_on_texts(X)
        self.num_words = min(self.max_vocab_size, len(self.tokenizer.word_index)+1)
        sequences = self.tokenizer.texts_to_sequences(X)

        #Defining the max sequence lenght
        if self.max_sequence_lenght is not None: 
            if not isinstance(self.max_sequence_lenght, (int, str)): 
                raise ValueError('Ivalid value for max_sequence_lenght. Please use an interger value or the strings "max" or "average".')
        else: 
            self.max_sequence_lenght = 'max'

        if isinstance(self.max_sequence_lenght, (str)): 
            self.max_sequence_lenght = self.get_max_sequence_lenght(sequences,self.max_sequence_lenght)
        
        sequences = pad_sequences(sequences,maxlen=self.max_sequence_lenght)
        return sequences 


    def transform(self, X): 
        sequences = self.tokenizer.texts_to_sequences(X)
        sequences = pad_sequences(sequences,maxlen=self.max_sequence_lenght)
        return sequences 

    def get_max_sequence_lenght(self, sequences,max_sequence_lenght): 
        function = None 
        if max_sequence_lenght not in ['max','average']: 
            raise ValueError('Ivalid value for max_sequence_lenght. Please use an interger value or the strings "max" or "average".')
        else: 
            function = max if max_sequence_lenght == 'max' else mean 
        
        return int(function(len(s) for s in sequences))