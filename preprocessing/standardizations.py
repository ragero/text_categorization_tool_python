import numpy as np 

class NormStandardization(object):

    def fit(self, data):
        self.standardization(data)

    def fit_transform(self, data):
        self.standardization(data)

    def transform(self, data):
        self.standardization(data)

    def standardization(self, data):
        for i in range(len(data)):
            data[i] = data[i] / np.linalg.norm(data[i])
        return data



