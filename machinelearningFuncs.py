import math
import numpy as np
import sklearn
import random


from sklearn import model_selection

class MachineLearning:
    def __init__(self, dataframe, target):
        self.dataframe = dataframe
        self.target = self.dataframe.loc[:, target]
        self.features = self.dataframe.drop(target)

    def test_train_split(self, prop_test):
        self.train, self.test = model_selection.train_test_split(self.dataframe, test_size = prop_test)
    
    def k_means_cluster(self, k = None, error = 0.01, features = list(self.dataframe.columns)):

        min_max = np.array([])
        for i in features:
            x_min, x_max = math.floor(self.dataframe[:,i]), math.ceil(self.dataframe[:,i])
            min_max = np.append([[x_min, x_max]])
        
        clusters = np.array([])
        for i in range(0, k):
            center = [random.randrange(x1_min,  x1_max), random.randrange(x2_min,  x2_max)]
            clusters = clusters.append(center)

