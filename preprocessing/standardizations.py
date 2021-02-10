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
            norm = np.linalg.norm(data[i])
            data[i] = (data[i] / norm) if norm > 0 else 0
        return data

class SumStandardization(object):

    def fit(self, data):
        return self.standardization(data)

    def fit_transform(self, data):
        return self.standardization(data)

    def transform(self, data):
        return self.standardization(data)

    def standardization(self, data):
        for i in range(len(data)):
            sum = data[i].sum()
            data[i] = (data[i] / sum) if sum > 0 else 0
        return data



