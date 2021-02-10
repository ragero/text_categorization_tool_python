# %% [markdown]
# # Libaries
# Preprocessing Algorithms

from sklearn.feature_extraction.text import TfidfVectorizer

from os import path
import sys
sys.path.append(path.join(path.dirname(__file__), '..'))
from preprocessing.standardizations import NormStandardization
from preprocessing.standardizations import SumStandardization

# %% [markdown]
# # Definitions
preprocessors = {}
preprocessors['TfidfVectorizer'] = TfidfVectorizer
preprocessors['NormStandardization'] = NormStandardization
preprocessors['SumStandardization'] = SumStandardization