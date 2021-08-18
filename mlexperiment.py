import pandas as pd
import os
import numpy as np
import src.samplingFuncs as sm
import src.MLFuncs as ml

#Initialise repositories
_projroot = os.path.abspath('.')
_datadir = os.path.join(_projroot, 'data')
_preprocesseddir = os.path.join(_datadir, 'preprocesseddata')
_experimentdir = os.path.join(_datadir, 'experiments')
_rawdir = os.path.join(_datadir, 'rawdata')
_src = os.path.join(_projroot,'src')
_sampling = os.path.join(_experimentdir,'sampling')
_combsampling = os.path.join(_sampling, 'combsampling')
_classification = os.path.join(_experimentdir, "classification")
models = os.path.join(_classification, "models")

os.makedirs(os.path.dirname(models), exist_ok=True)



x_test = pd.read_excel(os.path.join(_combsampling, 'combsampling_features.xlsx'), sheet_name='test_data')
y_test = pd.read_excel(os.path.join(_combsampling, 'combsampling_target.xlsx'), sheet_name='test_data')['Target']
x_train = pd.read_excel(os.path.join(_combsampling, 'combsampling_features.xlsx'), sheet_name='Smote_NC_4')
y_train = pd.read_excel(os.path.join(_combsampling, 'combsampling_target.xlsx'), sheet_name='Smote_NC_4')['Target']

ML = ml.machine_learning(x_train,x_test,y_train,y_test, _classification)
ML.SVM()
ML.boosting()
ML.perceptron()
ML.logistic_regression()
ML.random_forest()

ML.test_model()
ML.save_models('models')
ML.save_results("ML_funcs")
