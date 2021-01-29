import numpy as np 

class NormStandardization(object):

    def fit(self, data):
        return self.standardization(data)

    def fit_transform(self, data):
        return self.standardization(data)

    def transform(self, data):
        return self.standardization(data)

    def standardization(self, data):
        for i in range(len(data)):
            data[i] = data[i] / np.linalg.norm(data[i])
        return data

class MaxStandardization(object):

    def fit(self, data):
        return self.standardization(data)

    def fit_transform(self, data):
        return self.standardization(data)

    def transform(self, data):
        return self.standardization(data)

    def standardization(self, data):
        for i in range(len(data)):
            data[i] = data[i] / data[i].max()
        return data



