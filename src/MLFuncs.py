import pandas as pd
import os
from openpyxl import Workbook
import numpy as np
from collections import Counter
import pylab as pl
import joblib
import warnings
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

warnings.filterwarnings('ignore', category = FutureWarning)
warnings.filterwarnings('ignore', category = FutureWarning)

class machine_learning(self):

    def __init__(self, path):
        self.dataframe = {}
        self.target = {}
        self.features = {}
        self.results= {}

        if os.path.isdir(path):
            for filename in os.listdir(path):
                if filename.endswith('.csv'):
                    self.dataframe[filename] = pd.read_csv(os.path.join(path, filename))
                    try:
                        self.target[filename] = self.dataframe[filename]['Target']
                        self.features[filename] = self.dataframe[filename].drop('Target', axis = 1) 
                    except KeyError:
                        print('No target found')
        else:
            filename = os.path.basename(path)
            self.dataframe[filename] = pd.read_csv(path)

            try:
                self.target[filename] = self.dataframe[filename]['Target']
                self.features[filename] = self.dataframe[filename].drop('Target', axis = 1) 
            except KeyError:
                print('No target found')

        return

    def save_results(self, filename):
        wb = Workbook()
        wb.save(filename = file)

        for key in self.results:
            df = pd.DataFrame.from_dict(self.results[key])
            with pd.ExcelWriter(file,engine="openpyxl", mode = 'a') as writer:
                df.to_excel(writer, sheet_name = key)
            
        return

    def SVM(self):
        pass

    def perceptron(self):
        pass

    def random_forest(self):
        pass

    def boosting(self):
        pass

    def logistic_regression(self, C = [0.001,0.01,0.1,1,10,1000]): 
        lr = LogisticRegression()
        params = {
            'C' : C
        }
        lr_CV = GridSearchCV(lr)
        lr_CV.fit(self.features, self.target)
        self

    def time_func(self, func):
        pass
