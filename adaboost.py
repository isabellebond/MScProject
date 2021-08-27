import pandas as pd
import os
import numpy as np
import src.adaboostFuncs as ad

#Initialise repositories
_projroot = os.path.abspath('.')
_datadir = os.path.join(_projroot, 'data')
_preprocesseddir = os.path.join(_datadir, 'preprocesseddata')
_experimentdir = os.path.join(_datadir, 'experiments')
_rawdir = os.path.join(_datadir, 'rawdata')
_src = os.path.join(_projroot,'src')
_sampling = os.path.join(_experimentdir,'sampling')

splitsampling = os.path.join(_sampling,'splitsampling')
adaboost = os.path.join(_experimentdir,'adaboosting')

if not os.path.exists(adaboost):
    os.makedirs(adaboost)

def drop_unnamed(dataframe):
    try:
        dataframe = dataframe.drop("Unnamed: 0", axis = 1)
    except KeyError:
        pass
    
    return dataframe

features = pd.read_excel(os.path.join(splitsampling, 'splitsampling_features.xlsx'), sheet_name = 'train_test_split')
features = drop_unnamed(features)

target = pd.read_excel(os.path.join(splitsampling, 'splitsampling_target.xlsx'), sheet_name = 'train_test_split')['Target']

features_test = pd.read_excel(os.path.join(splitsampling, 'splitsampling_features.xlsx'), sheet_name = 'test_data')
features_test = drop_unnamed(features_test)

target_test = pd.read_excel(os.path.join(splitsampling, 'splitsampling_target.xlsx'), sheet_name = 'test_data')['Target']

ada = ad.Ada_Boost(features, target, features_test, target_test)
ada.data_split()
ada.model_creation(params = {'C' :[0.001,0.01,0.1,1,10,1000]})
ada.ada_boost(4, 'quarter_data')
y = ada.ada_predict()
ada.find_model_metrics(y, 'four_split')
ada.save_data(adaboost, 'four_split')

ada = ad.Ada_Boost(features, target, features_test, target_test)
ada.data_split(ratio=0.5, dataOut = 'half_data')
ada.model_creation(params = {'C' :[0.001,0.01,0.1,1,10,1000]})
ada.ada_boost(2, 'half_data')
y = ada.ada_predict()
ada.find_model_metrics(y, 'two_split')
ada.save_data(adaboost, 'two_split')

ada = ad.Ada_Boost(features, target, features_test, target_test)
ada.data_split(ratio=0.125, dataOut = 'eigth_data')
ada.model_creation(params = {'C' :[0.001,0.01,0.1,1,10,1000]})
ada.ada_boost(2, 'eigth_data')
y = ada.ada_predict()
ada.find_model_metrics(y, 'eight_split')
ada.save_data(adaboost, 'eight_split')