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
from sklearn.metrics import f1_score, recall_score, roc_auc_score, brier_score_loss

warnings.filterwarnings('ignore', category = FutureWarning)
warnings.filterwarnings('ignore', category = FutureWarning)

class machine_learning():

    def __init__(self, X_train, X_test, y_train, y_test, directory):
        self.results = {}
        self.models = {}
        self.metrics = {}
        
        self.features = X_train
        self.target= y_train
        self.X_test = X_test
        self.y_test = y_test

        self.dir = directory

        return

    def impute(self):
        imp = IterativeImputer(max_iter=10, random_state=0)

        self.features = imp.fit_transform(self.features)
        
        return 

    def save_results(self, prefix):

        wb = Workbook()
        wb.save(filename = os.path.join(self.dir, '%s_creation_metrics.xlsx'%prefix))

        for key in self.results:
            df = pd.DataFrame.from_dict(self.results[key])
            with pd.ExcelWriter(os.path.join(self.dir, '%s_creation_metrics.xlsx'%prefix), engine="openpyxl", mode = 'a') as writer:
                df.to_excel(writer, sheet_name = key)

        dataframe = pd.DataFrame.from_dict(self.metrics)    
        dataframe.to_excel(os.path.join(self.dir, '%s_model_metrics.xlsx'%prefix))
            
        return

    def save_models(self, directory):

        path = os.path.join(self.dir, directory)

        if not os.path.exists(path):
            os.makedirs(path)
        print(path)
        for key in self.models:
            joblib.dump(self.models[key], os.path.join(path, "%s_model.pkl"%key ))
        return

    def SVM(self, C = [0.001,0.1,1,10,100,1000], kernel = ['linear', 'rbf'], max_iters = 10000):
        svc = SVC(max_iter = max_iters)
        params = {
            'kernel': kernel,
            'C': C
        }
        svm_CV = GridSearchCV(svc, params, cv=5)
        svm_CV.fit(self.features, self.target)
        
        self.results['SVM'] = svm_CV.cv_results_
        self.models['SVM'] = svm_CV.best_estimator_

        return

    def perceptron(self, hidden_layers = [(10,),(50,),(100,)], activation = ['relu','tanh','logistic'], learning_rate = ['constant', 'invscaling','adaptive'], max_iters = 10000):
        mlp = MLPClassifier(max_iter = max_iters)
        print(mlp.get_params().keys())
        params = {
            'hidden_layer_sizes': hidden_layers,
            'activation' : activation,
            'learning_rate' : learning_rate
        }

        mlp_CV = GridSearchCV(mlp, params, cv=5)
        mlp_CV.fit(self.features, self.target)

        self.results['multi-layer_perceptron'] = mlp_CV.cv_results_
        self.models['multi-layer_perceptron'] = mlp_CV.best_estimator_

        return

    def random_forest(self, estimators = [5,50,250], depth=[2, 4, 8, 16, 32, None],max_iters = 10000):
        rf = RandomForestClassifier()
        params = {
            'n_estimators': estimators,
            'max_depth': depth
        }

        rf_CV = GridSearchCV(rf, params, cv=5)
        rf_CV.fit(self.features, self.target)

        self.results['random_forest'] = rf_CV.cv_results_
        self.models['random_forest'] = rf_CV.best_estimator_

        return

    def boosting(self, estimators = [5,50,250,500], depth =[1,3,5,7,9], learning_rate=[0.01,0.1,1,10,100],max_iters = 10000):
        gb = GradientBoostingClassifier()
        params = {
            'n_estimators': estimators,
            'max_depth': depth,
            'learning_rate': learning_rate
        }
        gb_CV = GridSearchCV(gb, params, cv=5)
        gb_CV.fit(self.features, self.target)
        
        self.results['gradient_boosting'] = gb_CV.cv_results_
        self.models['gradient_boosting'] = gb_CV.best_estimator_

        return

    def logistic_regression(self, C = [0.001,0.01,0.1,1,10,1000],max_iters = 10000): 
        lr = LogisticRegression()
        params = {
            'C' : C
        }
        lr_CV = GridSearchCV(lr, params, cv=5)
        lr_CV.fit(self.features, self.target)
        
        self.results['logistic_regression'] = lr_CV.cv_results_
        self.models['logistic_regression'] = lr_CV.best_estimator_

        return

    def test_model(self):

        for key in self.models:
            lr = self.models[key].fit(self.features, self.target)
            y_pred_test = lr.predict(self.X_test)
            y_test = self.y_test
            y_pred_train = lr.predict(self.features)
            y_train = self.target

            metrics = {}
            metrics['roc_auc_test'] = roc_auc_score(y_test, y_pred_test)
            metrics['roc_auc_train'] = roc_auc_score(y_train, y_pred_train)
            metrics['f1_test'] = f1_score(y_test, y_pred_test)
            metrics['f1_train'] = f1_score(y_train, y_pred_train)
            metrics['recall_test'] = recall_score(y_test, y_pred_test)
            metrics['recall_train'] = recall_score(y_train, y_pred_train)
            metrics['brier_score'] = brier_score_loss(y_train, y_pred_train)
            metrics['brier_score'] = brier_score_loss(y_train, y_pred_train)

            self.metrics[key] = metrics
            print(self.metrics)
        
        return
