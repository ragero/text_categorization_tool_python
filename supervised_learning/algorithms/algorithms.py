# Algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

dict_algorithms = {}
dict_algorithms['KNeighborsClassifier'] = KNeighborsClassifier
dict_algorithms['MultinomialNB'] = MultinomialNB
dict_algorithms['SVC'] = SVC
dict_algorithms['LogisticRegression'] = LogisticRegression
dict_algorithms['DecisionTreeClassifier'] = DecisionTreeClassifier