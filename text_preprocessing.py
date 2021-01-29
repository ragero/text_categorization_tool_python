# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Libraries

# %%
#Libraries to manage text data 

## SKLearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

## NLTK
from nltk.stem import RSLPStemmer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('rslp')
stemmer_pt = RSLPStemmer()
stemmer_en = PorterStemmer()
lemmatizer = WordNetLemmatizer()
from nltk.tokenize import word_tokenize

## Python
import string

## Gensim
import gensim
from gensim.models import KeyedVectors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

#Libraries to manage the file system
import os

#Other libraries
from tqdm import tqdm
import numpy as np
import scipy
import joblib
from abc import ABCMeta, abstractmethod
import json
import pandas as pd
import sys

# %% [markdown]
# # Loading the Stopwords

# %%
stopwords_pt = set({})
stopwords_en = set({})
path_stop_pt = './stopPort.txt'
path_stop_en = './stopIngl.txt'

if(os.path.exists(path_stop_pt) and os.path.exists(path_stop_en)): 
    with open(path_stop_pt) as file_stop_pt:
        for line in file_stop_pt.readlines():
            stopwords_pt.add(line.strip())
    with open(path_stop_en) as file_stop_en:
        for line in file_stop_en.readlines():
            stopwords_en.add(line.strip())
else: 
    stopwords_pt = set(stopwords.words('portuguese'))
    stopwords_en = set(stopwords.words('english'))

# %% [markdown]
# # Class to Tokenize and Clean the Texts

# %%
class TextPreprocessor(object): 
    
    def __init__(self, language='en', remove_stopwords=True, remove_punctuation=True, 
                 convert_numbers = True, remove_numbers = False, simplification=True, 
                 simplification_type='lemmatization', lower_case = True): 
        self.language = language
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.convert_numbers = convert_numbers
        self.remove_numbers = remove_numbers
        self.simplification = simplification
        self.simplification_type = simplification_type 
        self.lower_case = lower_case


    # Complete function to standardize the text
    def text_cleaner(self, text): 
        new_text = ''
        stopwords = None 

        if self.language == 'en':
            stopwords = stopwords_en 
        else:
            stopwords = stopwords_pt

        if self.lower_case == True: 
            text = text.lower()

        tokens = nltk.word_tokenize(text)
        
        if self.remove_stopwords == True:
            new_tokens = []
            for token in tokens: 
                if token in stopwords:
                    continue 
                else: 
                    new_tokens.append(token)
            tokens = new_tokens 

        if self.remove_punctuation == True: 
            new_tokens = []
            for token in tokens: 
                if token in string.punctuation:
                    continue 
                else: 
                    new_tokens.append(token)
            tokens = new_tokens 
        
        if self.remove_numbers == True:
            new_tokens = []
            for token in tokens: 
                if token.isnumeric():
                    continue
                new_tokens.append(token)
            tokens = new_tokens 
        
        if self.convert_numbers == True: 
            new_tokens = []
            for token in tokens: 
                if token.isnumeric():
                    new_tokens.append("0"*len(token))
                else: 
                    new_tokens.append(token)
            tokens = new_tokens 

        if self.simplification == True: 
            new_tokens = []
            if self.language == 'en': 
                if self.simplification_type  == 'lemmatization':
                    for token in tokens: 
                        new_tokens.append(lemmatizer.lemmatize(token))
                elif self.simplification_type  == 'stemming':
                    for token in tokens: 
                        new_tokens.append(stemmer_en.stem(token))
                else: 
                    raise ValueError('Unsuported language. Please, use language = {"pt","en"}.')
            elif self.language == 'pt':
                for token in tokens: 
                        new_tokens.append(stemmer_en.stem(token))
            else: 
                raise ValueError('Unsuported language. Please, use language = {"pt","en"}.')
            tokens = new_tokens

        return ' '.join(tokens).strip()


    #Just a simple tokenizer
    def tokenizer(self, text):
        text = text.lower()
        lista_alfanumerica = []

        for token in nltk.word_tokenize(text):
            if token in string.punctuation:
                continue 
            if token in stopwords_en: 
                continue
            if token.isnumeric():
                token = "0"*len(token)

            lista_alfanumerica.append(token)
        return lista_alfanumerica

# %% [markdown]
# # Functions to Save and Load the Presentations

# %%
def save_representation(representation, path): 
        joblib.dump(representation,path)
        
def load_representation(path): 
    return joblib.load(path)

# %% [markdown]
# # Class to Generate a Standard Representation for Different Space Vector Models

# %%
class StructuredRepresentation():

    def __init__(self, doc_vectors=None, class_vectors=None, vocabulary=None): 
        self.text_vectors = doc_vectors
        self.class_vectors = class_vectors 
        self.vocabulary = vocabulary

  
    def save_arff(self, name, path, non_sparse_format = False):
        num_docs = self.text_vectors.shape[0]
        num_attrs = self.text_vectors.shape[1]
        with open(path, 'w') as arff: 
            #Writting the relation
            arff.write(f'@relation {name}\n\n')
            
            #Writting the attributes
            if self.vocabulary == None: 
                for attr in range(num_attrs): 
                    arff.write(f'@ATTRIBUTE dim{attr + 1} NUMERIC\n')
            else: 
                sorted_vocabulary = sorted(self.vocabulary.items(), key=lambda x: x[1])
                for attr in range(num_attrs): 
                    arff.write(f'@ATTRIBUTE {sorted_vocabulary[attr][0]} NUMERIC\n')
            
            #Writting the class names
            arff.write('@ATTRIBUTE att_class ' + '{"' + '","'.join(self.class_vectors.unique()) + '"}\n\n')


            #Writting the data
            arff.write('@data\n\n')


            if non_sparse_format == False: 
                for doc in range(num_docs):
                    vector = self.text_vectors[doc]
                    if type(vector) == scipy.sparse.csr.csr_matrix: 
                        vector = self.text_vectors[doc].toarray()[0]
                    str_vec = ''
                    for i in range(vector.shape[0]): 
                        str_vec += str(vector[i]) + ','
                    classe = self.class_vectors.iloc[doc]
                    arff.write(str_vec + '"' + classe + '"\n') 
            else: 
                for doc in range(num_docs):
                    vector = self.text_vectors[doc]
                    if type(vector) == scipy.sparse.csr.csr_matrix: 
                        vector = self.text_vectors[doc].toarray()[0]
                    str_vec = ''
                    for i in range(vector.shape[0]): 
                        if vector[i] > 0: 
                            str_vec += f'{i} {str(vector[i])},'
                    classe = self.class_vectors.iloc[doc]
                    arff.write('{' + str_vec + str(num_attrs) + ' "' + classe + '"}\n') 
    

# %% [markdown]
# # Classes to Generate Vector Space Model Based Representaions 
# %% [markdown]
# ## Bag-of-Words or Bag-of-N-Grams

# %%
class MySparseVSM: 

    def __init__(self, weight='tf', n_grams=1):
        self.vectorizer = None 
        if(weight == 'tf'):
            self.vectorizer = CountVectorizer(min_df=2, ngram_range=(1, n_grams), dtype=np.uint8)
        else:
            self.vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, n_grams), dtype=np.uint8)

        self.structured_representation = None

    def build_representation(self, texts, classes): 
        self.structured_representation = StructuredRepresentation(self.vectorizer.fit_transform(texts), classes, self.vectorizer.vocabulary_)
        return self.structured_representation

# %% [markdown]
# ## Low Dimensional Representations
# %% [markdown]
# ### SuperClass

# %%
class LowDimensionalRepresentation(object):
    
    def __init__(self, dim_size = 200, model = None, num_threads=1, min_count = 2, window_size = 5, num_max_epochs= 100, alpha = 0.025, min_alpha = 0.0001): 
        __metaclass__  = ABCMeta
        self.dim_size = dim_size
        self.model = model 
        self.num_threads = num_threads 
        self.min_count = min_count
        self.window_size = window_size
        self.num_max_epochs = num_max_epochs
        self.alpha = alpha
        self.min_alpha = min_alpha

    @abstractmethod
    def build_representation(self, texts, classes): 
        pass    


# %%
class MyWord2Vec (LowDimensionalRepresentation):
    
    def __init__(self, dim_size = 200, model = 'skip-gram', method='average', num_threads=1, min_count = 2, window_size = 5, num_max_epochs = 100, alpha = 0.025, min_alpha = 0.0001): 
        super(MyWord2Vec,self).__init__(dim_size,model,num_threads,min_count,window_size, num_max_epochs, alpha, min_alpha)
        self.language_model = None 
        self.cg = None 
        if method != 'average' and method != 'sum': 
            raise ValueError('Unsuported method. Please, use method = {"average","sum"}.')
        self.method = method
        

    def build_model(self, texts):
        language_model = None
        sg = 0
        if self.model == 'cbow' : 
            language_model = gensim.models.Word2Vec
        elif self.model == 'sg': 
            language_model = gensim.models.Word2Vec
            sg = 1
        #elif self.model == 'glove': 
            #self.language_model = gensim.models.Word2Vec(list_tokens_texts,min_count=min_count,window=window_size, size=dim_size, workers=num_threads)
        elif self.model == 'fasttext': 
            language_model = gensim.models.FastText
            sg = 1
        else: 
            raise ValueError('Unsuported language model. Please, use language model = {"cbow","sg","fasttext"}.')

        list_tokens_texts = texts.apply(self.tokenizer)
        self.language_model = language_model(list_tokens_texts,sg=sg, min_count=self.min_count,
                                             window=self.window_size, size=self.dim_size, workers=self.num_threads, 
                                             iter=self.num_max_epochs, alpha = self.alpha, min_alpha = self.min_alpha)

    def build_representation(self, texts, classes): 
        self.build_model(texts)
        matrix = np.zeros((len(texts),self.dim_size))

        for i in range(len(texts)):
            tokens = self.tokenizer(texts.iloc[i])
            matrix[i] = self.sum_vectors(tokens)
            if(self.method == 'average' and len(tokens) > 0): 
                matrix[i] = matrix[i]/len(tokens)


        self.structured_representation = StructuredRepresentation(matrix, classes, None)
        return self.structured_representation

    def tokenizer(self,text):
        text = text.lower()
        lista_alfanumerica = []

        for token in nltk.word_tokenize(text):
            if token in string.punctuation:
                continue 
            if token in stopwords_en: 
                continue
            if token.isnumeric():
                token = "0"*len(token)

            lista_alfanumerica.append(token)
        return lista_alfanumerica

    def sum_vectors(self,lista_tokens): 
        vetor_combinado = np.zeros(self.dim_size)
        for token in lista_tokens: 
            try:
                vetor_combinado += self.language_model.wv.get_vector(token)
            except KeyError:
                continue 
        return vetor_combinado


# %%
class MyDoc2Vec (LowDimensionalRepresentation):
    
    def __init__(self, dim_size = 200, model = 'dm', method='average', num_threads=4, alpha = 0.025, min_alpha=0.0001, num_max_epochs = 2000,min_count = 1, window_size = 5): 
        super(MyDoc2Vec,self).__init__(dim_size,model,num_threads,min_count,window_size, num_max_epochs, alpha, min_alpha)
        self.model = model
      

        self.dm = -1
        if model == 'dbow':
            self.dm = 0
        elif model == 'dm':
            self.dm = 1
        elif model != 'both':
            raise ValueError('Unsuported model. Please, use model = {"dm","dbow"}.')
        
        self.dm_mean = 1
        if method == 'average': 
            self.dm_concat = 0
        elif method == 'concat':
            self.dm_concat = 1
        else:
            raise ValueError('Unsuported method. Please, use method = {"concat","average"}.')
        
        #standard parameters
        self.hs = 0
        self.dbow_words = 0
        
    
    def build_model(self, texts): 
        
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(texts)]
        if self.model == 'dm' or self.model == 'dbow': 
            model = Doc2Vec(vector_size=self.dim_size, alpha=self.alpha, min_alpha=self.min_alpha, 
                            min_count=self.min_count, dm=self.dm, workers = self.num_threads,
                            dm_min = self.dm_mean, dm_concat = self.dm_concat,
                            dbow_words = self.dbow_words, hs=self.hs, epochs=self.num_max_epochs, seed=1)
            model.build_vocab(tagged_data)
            model.train(tagged_data,total_examples=model.corpus_count,epochs=model.iter)
            
            #Reduce mnanemory usage
            model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

            matrix = np.zeros((len(texts),self.dim_size))
            for i in range(model.corpus_count): 
                matrix[i] = model.docvecs[str(i)]
            return matrix
        elif self.model == 'both': 
            modelDM = Doc2Vec(vector_size=self.dim_size, alpha=self.alpha, min_alpha=self.min_alpha, 
                            min_count=self.min_count, dm=1, workers = self.num_threads,
                            dm_min = self.dm_mean, dm_concat = self.dm_concat,
                            dbow_words = self.dbow_words, hs=self.hs, epochs=self.num_max_epochs, seed=1)
            modelDBOW = Doc2Vec(vector_size=self.dim_size, alpha=self.alpha, min_alpha=self.min_alpha, 
                            min_count=self.min_count, dm=0, workers = self.num_threads,
                            dm_min = self.dm_mean, dm_concat = self.dm_concat,
                            dbow_words = self.dbow_words, hs=self.hs, epochs=self.num_max_epochs, seed=1)
                        
            modelDM.build_vocab(tagged_data)
            modelDBOW.build_vocab(tagged_data)

            modelDM.train(tagged_data,total_examples=modelDM.corpus_count,epochs=modelDM.iter)
            modelDBOW.train(tagged_data,total_examples=modelDBOW.corpus_count,epochs=modelDBOW.iter)
            
            #Reduce memory usage
            modelDM.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
            modelDBOW.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)


            matrixDM = np.zeros((len(texts),self.dim_size))
            for i in range(modelDM.corpus_count): 
                matrixDM[i] = modelDM.docvecs[str(i)]
            matrixDBOW = np.zeros((len(texts),self.dim_size))
            for i in range(modelDBOW.corpus_count): 
                matrixDBOW[i] = modelDBOW.docvecs[str(i)]
            
            matrix = np.concatenate([matrixDM, matrixDBOW], axis=1)
            return matrix

  

    def build_representation(self, texts, classes): 
        self.structured_representation = StructuredRepresentation(self.build_model(texts), classes, None)
        return self.structured_representation

# %% [markdown]
# ## Based on Word Embeddings
# %% [markdown]
# # Área de Testes

# %%
# Criando um dicionario (versão completa)
config = {}
config['csvs_diretory'] = '/home/rafael/Área de Trabalho/Temp/Teste/Entrada'
config['output_directory'] = '/media/rafael/DadosCompartilhados/Representacoes/Word2Vec/CSTR'
config['text_column'] = 'text'
config['class_column'] = 'class'
config['pre-processing'] = True
config['pre-processing_steps'] = {'language' : 'en', 'remove_stopwords' : True, 'remove_punctuation' : True, 
                 'convert_numbers' : True, 'remove_numbers' : True, 'simplification' : True, 
                 'simplification_type' : 'lemmatization', 'lower_case' : True}
config['sparse_representation'] = {'use': False, 'n-grams' : [1], 'term-weights' : ['tf', 'tf-idf']}
config['low-dimension_representation'] = {'use' : True, 'types' : ['doc2vec', 'word2vec'] ,
                                          'doc2vec_config' : {'use' : False, 'models' : ['dm', 'dbow', 'both'], 'methods' : ['average','concat'], 
                                          'num_threads': 4, 'alpha' : 0.025, 'min_alpha' : 0.001,
                                          'num_max_epochs' : [1, 3, 100, 1000], 'min_count' : 1, 'window_sizes' : [5, 8, 10], 
                                          'num_dimensions' : [25,  50, 100, 500, 1000] }, 
                                          'word2vec_config' : {'use' : True, 'models' : ['sg','cbow','fasttext'], 'methods' : ['average','sum'], 
                                          'num_threads': 4, 'alpha' : 0.025, 'min_alpha' : 0.0001,                       
                                          'num_max_epochs' : [1, 3, 5, 50, 100], 'min_count' : 5, 'window_sizes' : [5, 8, 10], 
                                          'num_dimensions' : [25,  50, 100, 300] }
                                         }
config['save-arff'] = True 
config['save-binary'] = False


# %%
# Saving the Json
def save_json(path_json): 
    with open(path_json, 'w') as outfile:
        json.dump(config, outfile, indent=4, ensure_ascii=False,)


# %%
# Loading the Json
def load_json(path_json): 
    with open(path_json) as json_file:
        return json.load(json_file)


# %%
def load_csv(path, text_column, class_column): 
    df = pd.read_csv(path)
    df = df.dropna()
    texts = df[config['text_column']]
    classes = df[config['class_column']]
    return texts, classes 


# %%
def build_name(name, representation_type, config): 
    final_name = f'{name}_{representation_type}'
    for item in config.items(): 
        final_name += f'_{item[0]}={item[1]}'
    
    return final_name


# %%
def build_and_save_representation(config, rep_builder, texts, classes, name_builder, parameters, dataset_name, non_sparse_format): 
    representation_name = build_name(dataset_name, name_builder, parameters)
    path_out_arff = os.path.join(config['output_directory'], representation_name + '.arff') 
    path_out_bin = os.path.join(config['output_directory'], representation_name + '.rep') 

    if os.path.exists(path_out_arff) or os.path.exists(path_out_bin): 
        return 

    representation = rep_builder.build_representation(texts,classes)

    if config['save-arff'] == True: 
        representation.save_arff(representation_name, path_out_arff, non_sparse_format = non_sparse_format)
    if config['save-binary'] == True: 
        save_representation(representation, path_out_bin)


# %%



# %%
def process_all(config): 

    #Getting the directory of the csvs and listing the csvs 
    text_preprocessor = TextPreprocessor(**config['pre-processing_steps'])
    directory = config['csvs_diretory']
    for csv_file in sorted(os.listdir(directory)):
        dataset_name = csv_file[:csv_file.rindex('.')]
        print('=============================================')
        print('=============================================')
        print('Dataset: ', dataset_name)
        
        # Loading the CSVs and getting the column of the texts and the classes
        texts, classes = load_csv(os.path.join(directory,csv_file), config['text_column'], config['class_column'])

        #Pre-prossing texts
        if config['pre-processing'] == True: 
            print('Preprocessing text collection')
            texts = texts.apply(text_preprocessor.text_cleaner)
        
        #Processing sparse representations
        if config['sparse_representation']['use'] == True: 
            print('=============================================')
            print('Sparse Representation')
            for ngram in config['sparse_representation']['n-grams']: 
                print('N-gram: ', ngram)
                for term_weight in config['sparse_representation']['term-weights']: 
                    print('Term-weight: ', term_weight)
                    parameters = {'term-weight' : term_weight, 'n-grams' : ngram}
                    mySparseVSM = MySparseVSM(weight=term_weight, n_grams=ngram)
                    build_and_save_representation(config, mySparseVSM, texts, classes, 'SparseVSM' ,parameters, dataset_name, True)
                
        #Processing low-dimensional representations
        if config['low-dimension_representation']['use'] == True:
            for type_repr in config['low-dimension_representation']['types']: 
                if type_repr == 'doc2vec': 
                    if config['low-dimension_representation']['doc2vec_config']['use'] == True: 
                        print('=============================================')
                        print('Doc2Vec')
                        for model in config['low-dimension_representation']['doc2vec_config']['models']:
                            print('Model:', model)
                            for method in config['low-dimension_representation']['doc2vec_config']['methods']: 
                                print('Method:', method)
                                for num_max_epoch in config['low-dimension_representation']['doc2vec_config']['num_max_epochs']:
                                    print('Num. Max Epochs:', num_max_epoch)
                                    for window_size in config['low-dimension_representation']['doc2vec_config']['window_sizes']:
                                        print('Window Size:', window_size)
                                        for num_dimensions in config['low-dimension_representation']['doc2vec_config']['num_dimensions']:
                                            print('Num. Dimensions:', num_dimensions)
                                            parameters = {'model' : model, 'method' : method, 'dim_size': num_dimensions,
                                                        'num_max_epochs' : num_max_epoch, 'window_size' : window_size, 
                                                        'num_threads' : config['low-dimension_representation']['doc2vec_config']['num_threads'],
                                                        'min_count' : config['low-dimension_representation']['doc2vec_config']['min_count'],
                                                        'alpha' : config['low-dimension_representation']['doc2vec_config']['alpha'],
                                                        'min_alpha' : config['low-dimension_representation']['doc2vec_config']['min_alpha']
                                                        }

                                            myDoc2Vec = MyDoc2Vec(**parameters)
                                            build_and_save_representation(config, myDoc2Vec, texts, classes, 'Doc2Vec', parameters, dataset_name, False)
                elif type_repr == 'word2vec': 
                    if config['low-dimension_representation']['word2vec_config']['use'] == True:
                        print('=============================================')
                        print('Word2Vec')
                        for model in config['low-dimension_representation']['word2vec_config']['models']:
                            print('Model:', model)
                            for method in config['low-dimension_representation']['word2vec_config']['methods']: 
                                print('Method:', method)
                                for num_max_epoch in config['low-dimension_representation']['word2vec_config']['num_max_epochs']:
                                    print('Num. Max Epochs:', num_max_epoch)
                                    for window_size in config['low-dimension_representation']['word2vec_config']['window_sizes']:
                                        print('Window Size:', window_size)
                                        for num_dimensions in config['low-dimension_representation']['word2vec_config']['num_dimensions']:
                                            print('Num. Dimensions:', num_dimensions)
                                            parameters = {'model' : model, 'method' : method, 'dim_size': num_dimensions,
                                                        'num_max_epochs' : num_max_epoch, 'window_size' : window_size, 
                                                        'num_threads' : config['low-dimension_representation']['word2vec_config']['num_threads'],
                                                        'min_count' : config['low-dimension_representation']['word2vec_config']['min_count'],
                                                        'alpha' : config['low-dimension_representation']['word2vec_config']['alpha'],
                                                        'min_alpha' : config['low-dimension_representation']['word2vec_config']['min_alpha']
                                                        }

                                            myWord2Vec = MyWord2Vec(**parameters)
                                            build_and_save_representation(config, myWord2Vec, texts, classes, 'Word2Vec', parameters, dataset_name, False)
                    pass
                else: 
                    raise ValueError('Unsuported low dimension representation type. Please, use types = {"doc2vec","word2vec"}.')
                
                

    print('Process Concluded!!')
                

# %% [markdown]
# #  Main Function

# %%
if __name__ == '__main__': 
    path_json = sys.argv[1]
    print('Path JSON:', path_json)
    if not os.path.exists(path_json): 
        print('Incorrect path for JSON file')
        sys.exit(0)
    else: 
        extension = os.path.splitext(path_json)[1]
        print('Extension', extension)
        if extension.lower() != '.json': 
            print('Invalid extension file')
            sys.exit(0)
        else: 
            config = load_json(path_json)
            process_all(config)



