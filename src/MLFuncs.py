import pandas as pd
import os
from openpyxl import Workbook
import numpy as np
from collections import Counter
import pylab as pl
import joblib
import warnings
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

warnings.filterwarnings('ignore', category = FutureWarning)
warnings.filterwarnings('ignore', category = FutureWarning)

class machine_learning():

    def __init__(self, path):
        self.results = {}
        self.models = {}

        self.dataframe = pd.read_csv(path)

        try:
            self.target = self.dataframe['Target']
            self.features = self.dataframe.drop('Target', axis = 1) 
        except KeyError:
            print('No target found')

        return

    def impute(self):
        imp = IterativeImputer(max_iter=10, random_state=0)

        self.features = imp.fit_transform(self.features)
        
        return 

    def save_results(self, file):
        wb = Workbook()
        wb.save(filename = file)

        for key in self.results:
            df = pd.DataFrame.from_dict(self.results[key])
            with pd.ExcelWriter(file,engine="openpyxl", mode = 'a') as writer:
                df.to_excel(writer, sheet_name = key)
            
        return
    
    def save_models(self, directory):
        for key in self.models:
            joblib.dump(self.models[key], os.path.join(directory, "%s.pkl"%key))
        return

    def SVM(self, C = [0.001,0.1,1,10,100,1000], kernel = ['linear', 'rbf'], max_iters = 5000):
        svc = SVC(max_iter = max_iters)
        params = {
            'kernel': kernel,
            'C': C
        }
        svm_CV = GridSearchCV(svc, params, cv=5)
        svm_CV.fit(self.features, self.target)
        
        self.results['support vector machine'] = svm_CV.cv_results_
        self.models['support vector machine'] = svm_CV.best_estimator_

        return

    def perceptron(self, hidden_layers = [(10,),(50,),(100,)], activation = ['relu','tanh','logistic'], learning_rate = ['constant', 'invscaling','adaptive'], max_iters = 5000):
        mlp = MLPClassifier(max_iter = max_iters)
        print(mlp.get_params().keys())
        params = {
            'hidden_layer_sizes': hidden_layers,
            'activation' : activation,
            'learning_rate' : learning_rate
        }

        mlp_CV = GridSearchCV(mlp, params, cv=5)
        mlp_CV.fit(self.features, self.target)

        self.results['multi-layer perceptron'] = mlp_CV.cv_results_
        self.models['multi-layer perceptron'] = mlp_CV.best_estimator_

        return

    def random_forest(self, estimators = [5,50,250], depth=[2, 4, 8, 16, 32, None],max_iters = 5000):
        rf = RandomForestClassifier()
        params = {
            'n_estimators': estimators,
            'max_depth': depth
        }

        rf_CV = GridSearchCV(rf, params, cv=5)
        rf_CV.fit(self.features, self.target)

        self.results['random forest'] = rf_CV.cv_results_
        self.models['random forest'] = rf_CV.best_estimator_

        return

    def boosting(self, estimators = [5,50,250,500], depth =[1,3,5,7,9], learning_rate=[0.01,0.1,1,10,100],max_iters = 5000):
        gb = GradientBoostingClassifier()
        params = {
            'n_estimators': estimators,
            'max_depth': depth,
            'learning_rate': learning_rate
        }
        gb_CV = GridSearchCV(gb, params, cv=5)
        gb_CV.fit(self.features, self.target)
        
        self.results['gradient boosting'] = gb_CV.cv_results_
        self.models['gradient boosting'] = gb_CV.best_estimator_

        return

    def logistic_regression(self, C = [0.001,0.01,0.1,1,10,1000],max_iters = 5000): 
        lr = LogisticRegression()
        params = {
            'C' : C
        }
        lr_CV = GridSearchCV(lr, params, cv=5)
        lr_CV.fit(self.features, self.target)
        
        self.results['logistic regression'] = lr_CV.cv_results_
        self.models['logistic regression'] = lr_CV.best_estimator_

        return

    def time_func(self, func):
        pass
