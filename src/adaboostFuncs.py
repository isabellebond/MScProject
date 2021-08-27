
import pandas as pd
import os
import numpy as np
import json
from collections import Counter
import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from openpyxl import Workbook
from sklearn.svm import SVC
import pylab as pl
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score, accuracy_score

class Ada_Boost():
    def __init__(self, features, target, features_test, target_test):
        self.features = features
        self.target = target
        self.features_test = features_test
        self.target_test = target_test
        self.split_features = {}
        self.split_target = {}
        self.models = {}
        self.ada_models = {}
        self.ada_alphas = {}
        self.metrics = {}

    def create_dataset(self, X_negative_class, X_positive_class, dataOut):
        k = 1
        for item in X_negative_class:
            y_negative_class = np.zeros((len(item),1))
            y_positive_class = np.ones((len(X_positive_class),1))
            self.split_target['%s_%s'%(dataOut, k)] = np.append(y_negative_class, y_positive_class, axis = 0).ravel()
            X_new = np.append(item, X_positive_class, axis = 0)
            self.split_features['%s_%s'%(dataOut, k)] = pd.DataFrame(data = X_new, columns = self.features.columns)
            k+=1

    def data_split(self, ratio = 0.25, dataOut = 'quarter_data'):
        """
        Divides total features in majority class into four smaller sub datasets.
        Appends all features in minority class into new subsets of data
        Saves new datasets as CSV files 
        Outputs first 2 principle components onto graph and saves in images

            Parameters:
                dataframe
                filename(str): filename of output files (save in preprocesseddir)
        
            Return:
        """
        class1 = []
        class2 = []
        for i in range(0,len(self.target)):
            if self.target.iloc[i] == 0:
                class1.append(i)
            else:
                class2.append(i)

        X_class1 = self.features.iloc[class1,:]
        X_class2 = self.features.iloc[class2,:]
        

        if ratio == 0.125: 
            Xa,Xb = train_test_split(X_class1, test_size = 0.5)

            Xz, Xy = train_test_split(Xa, test_size = 0.5)
            Xx, Xw = train_test_split(Xb, test_size = 0.5)

            X1, X2 = train_test_split(Xz, test_size = 0.5)
            X3, X4 = train_test_split(Xy, test_size = 0.5)
            X5, X6 = train_test_split(Xx, test_size = 0.5)
            X7, X8 = train_test_split(Xw, test_size = 0.5)

            X = [X1,X2,X3,X4,X5,X6,X7,X8]
            
            self.create_dataset(X, X_class2, dataOut)

        elif ratio == 0.25:
            Xa,Xb = train_test_split(X_class1, test_size = 0.5)
            X1, X2 = train_test_split(Xa, test_size = 0.5)
            X3, X4 = train_test_split(Xb, test_size = 0.5)

            X = [X1,X2,X3,X4]
            
            self.create_dataset(X, X_class2, dataOut)
        
        elif ratio == 0.5:
            Xa,Xb = train_test_split(X_class1, test_size = 0.5)

            X = [Xa,Xb]
            
            self.create_dataset(X, X_class2, dataOut)
        
        else:
            raise ValueError('Unaccepted value for ratio. Ratio must be 0.125, 0.25 or 0.5.')

        return
    
    def model_creation(self, max_iters = 1000, classifier = 'logistic_regression', params = {}, split = True):
        if classifier == 'logistic_regression':
            algorithm = LogisticRegression(max_iter = max_iters)
        elif classifier == 'boosting':
            algorithm = GradientBoostingClassifier(max_iter = max_iters)
        elif classifier == 'random_forest':
            algorithm = RandomForestClassifier(max_iter = max_iters)
        elif classifier == 'SVM':
            algorithm = SVC(max_iter = max_iters)
        elif classifier == 'perceptron':
            algorithm = MLPClassifier(max_iter = max_iters)

        cv = GridSearchCV(algorithm, params, cv=5)

        if split == True:
            for key in self.split_features:
                cv.fit(self.split_features[key], self.split_target[key])
                self.models[key] = cv.best_estimator_
        else:
            cv.fit(self.features, self.target)
            self.base_model = cv.best_estimator_

        return

    def ada_boost(self, kmax, prefix, dataOut = 'ada_boost'):

        models = {}
        for key in self.models:
            if prefix in key:
                models[key] = self.models[key]
        print(models)
        
        target = np.where(self.target == 0, -1, self.target)
        W = np.ones(len(target))/len(target)
        alphas = {}
        min_models = {}

        for k in range(0, kmax):
            errors = {}
            for key in models:
                y_pred = models[key].predict(self.features)
                y_pred = np.where(y_pred == 0, -1, y_pred)
                error = 0
                W_tot = 0
               
                for i in range(0, len(target)):
                    W_tot = W_tot + W[i]
                    
                    if (target[i]+y_pred[i]) == 0:
                        error = error + W[i]
        
                errors[key] = error/W_tot
        
            min_val = min(errors.values())
            min_key = [k for k, v in errors.items() if v == min_val]
            min_model = models[min_key[0]]
            
            if errors[min_key[0]] < 0.5:
                alpha = 0.5*(np.log((1-errors[min_key[0]])/(errors[min_key[0]])))
                alphas[min_key[0]] = alpha
                min_models[min_key[0]] = min_model

                W_new = np.array([])
                for i in range(0, len(target)):
                    W_new = np.append(W_new,(W[i]*np.exp(-(alpha*target[i]*y_pred[i]))))
                W = W_new

                del models[min_key[0]]

        self.ada_models[dataOut] = min_models
        print(self.ada_models)
        self.ada_alphas[dataOut] = alphas
        print(self.ada_alphas)

        return self.ada_alphas, self.ada_models
    
    def enhanced_ada_boost(self, kmax, prefix, dataOut = 'ada_boost'):

        models = {}
        for key in self.models:
            if prefix in key:
                models[key] = self.models[key]
        
        target = np.where(self.target == 0, -1, self.target)
        W = np.ones(len(target))/len(target)
        alphas = {}
        min_models = {}

        for k in range(0, kmax):
            errors = {}
            for key in models:
                y_pred = models[key].predict(self.features)
                y_pred = np.where(y_pred == 0, -1, y_pred)
                error = 0
                W_tot = 0
               
                for i in range(0, len(target)):
                    W_tot = W_tot + W[i]
                    
                    if (target[i]+y_pred[i]) == 0:
                        error = error + W[i]
        
                errors[key] = error/W_tot
        
            min_val = min(errors.values())
            min_key = [k for k, v in errors.items() if v == min_val]
            min_model = models[min_key[0]]

            if errors[min_key] < 0.5:
                alpha = 0.5*(np.log((1-errors[min_key[0]])/(errors[min_key[0]])))
                alphas[min_key[0]] = alpha
                min_models[min_key[0]] = min_model

                W_new = np.array([])
                for i in range(0, len(target)):
                    W_new = np.append(W_new,(W[i]*np.exp(-(alpha*target[i]*y_pred[i]))))
                W = W_new

                del models[min_key[0]]

        self.ada_models[dataOut] = min_models
        self.ada_alphas[dataOut] = alphas

        return self.ada_alphas, self.ada_models

    def ada_predict(self, train = True, ada_model = 'ada_boost'):
        
        if train == True:
            features = self.features
        else:
            features = self.features_test

        y = np.zeros(len(features))

        for key in self.ada_models[ada_model]:
            y_pred = self.ada_models[ada_model][key].predict(features)
            y_pred = np.where(y_pred == 0, -1, y_pred)
            print(y_pred)
            y_pred = y_pred * self.ada_alphas[ada_model][key]
            print(y_pred)
            y = y + y_pred
            print('y =', y)
        
        y = np.sign(y)
        print('y sign =', y)
        y = np.where(y == -1., 0, y)
        print('y fin', y)
        return y

    def find_model_metrics(self, y_pred, key, train = True,):

        if train == True:
            y_true = self.target
        else:
            y_true = self.target_test

        metrics = {}
        metrics['roc_auc_test'] = roc_auc_score(y_true, y_pred)
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['f1_test'] = f1_score(y_true, y_pred)
        metrics['recall_test'] = recall_score(y_true, y_pred)

        self.metrics[key] = metrics

        return
    
    def save_data(self, directory, prefix):

        path = os.path.join(directory, 'models')
        wb = Workbook()
        wb.save(filename = os.path.join(directory,'%s_features.xlsx'%prefix))
        wb.save(filename = os.path.join(directory,'%s_target.xlsx'%prefix))

        if not os.path.exists(path):
            os.makedirs(path)
        for key in self.models:
          
            try:
                df_features = self.features[key]
                df_target = self.target[key]
                with pd.ExcelWriter(os.path.join(directory,'%s_features.xlsx'%prefix), engine="openpyxl", mode = 'a') as writer:
                    df_features.to_excel(writer, sheet_name = key)
                with pd.ExcelWriter(os.path.join(directory,'%s_target.xlsx'%prefix), engine="openpyxl", mode = 'a') as writer:
                    df_target.to_excel(writer, sheet_name = key)
            except KeyError:
                pass
            joblib.dump(self.models[key], os.path.join(path, "%s.pkl"%key ))
            dataframe = pd.DataFrame.from_dict(self.metrics)    
            dataframe.to_excel(os.path.join(directory, '%s_model_metrics.xlsx'%prefix))
        for key in self.ada_models:
            try:
                df_features = self.split_features[key]
                df_target = self.split_target[key]
                with pd.ExcelWriter(os.path.join(directory,'%s_features.xlsx'%prefix), engine="openpyxl", mode = 'a') as writer:
                    df_features.to_excel(writer, sheet_name = key)
                with pd.ExcelWriter(os.path.join(directory,'%s_target.xlsx'%prefix), engine="openpyxl", mode = 'a') as writer:
                    df_target.to_excel(writer, sheet_name = key)
            except KeyError:
                pass
            joblib.dump(self.ada_models[key], os.path.join(path, "%s.pkl"%key ))
            dataframe = pd.DataFrame.from_dict(self.ada_alphas)    
            dataframe.to_excel(os.path.join(directory, '%s_adaboost_alphas.xlsx'%prefix))


    





