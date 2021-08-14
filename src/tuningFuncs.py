import pandas as pd
import os
from openpyxl import Workbook
import numpy as np
from collections import Counter
import pylab as pl
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#Initialise repositories
_projroot = os.path.abspath('.')
_datadir = os.path.join(_projroot, 'data')
_preprocesseddir = os.path.join(_datadir, 'preprocesseddata')
_rawdir = os.path.join(_datadir, 'rawdata')

class Tuning():
    def __init__(self, path):
        self.foldername = path
        self.dataframe = {}
        self.target = {}
        self.features = {}
        self.results = {}

    def read_csv(self, path):

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
    
    def impute(self):
        imp = IterativeImputer(max_iter=10, random_state=0)

        for key in self.features:
            self.features[key] = imp.fit_transform(self.features[key])
        
        return 
    def print_results(self):
        print('BEST PARAMS:{}\n'.format(results.best_params_))
        means = results.cv_results_['mean_test_score']
        stds = results.cv_results_['std_test_score']
        for mean, std, params in zip(means,stds, results.cv_results_['params']):
            print('{} (+/-{}) for {}'.format(round(mean, 3), round(std*2, 3),params))
        return
    
    def save_results(self, file):
        wb = Workbook()
        wb.save(filename = file)

        for key in self.results:
            df = pd.DataFrame.from_dict(self.results[key])
            with pd.ExcelWriter(file,engine="openpyxl", mode = 'a') as writer:
                df.to_excel(writer, sheet_name = key)
            
        return
    
    def elastic_regression(self, C = [0.001,0.01,1,10,100,1000], L1_ratio = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], tolerance = 0.1, max_iters = 100):
        C = np.array(C)
        L1_ratio = np.array(L1_ratio)
        for key in self.features:
            lr = LogisticRegressionCV(solver = 'saga',penalty = 'elasticnet', tol = tolerance, max_iter = max_iters)
            Cs = (np.multiply(np.ones((len(C),len(self.target[key]))), C[:,np.newaxis])).tolist()
            L1_ratios = (np.multiply(np.ones((len(L1_ratio),len(self.target[key]))), L1_ratio[:,np.newaxis])).tolist()
            print(Cs)
            parameters = {
                'Cs' : Cs,
                'l1_ratios' : L1_ratios
            }
            cv = GridSearchCV(lr, parameters, cv = 5)
            cv.fit(self.features[key], self.target[key])
            self.results[key] = cv.cv_results_
        print(self.results)
        return
        