import pandas as pd
import os
import numpy as np
import src.samplingFuncs as sm
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor

#Initialise repositories
_projroot = os.path.abspath('.')
_datadir = os.path.join(_projroot, 'data')
_preprocesseddir = os.path.join(_datadir, 'preprocesseddata')
_experimentdir = os.path.join(_datadir, 'experiments')
_rawdir = os.path.join(_datadir, 'rawdata')
_src = os.path.join(_projroot,'src')
_sampling = os.path.join(_experimentdir,'sampling')

undersampling = os.path.join(_sampling,'undersampling')
oversampling = os.path.join(_sampling,'oversampling')
combsampling = os.path.join(_sampling,'combsampling')
splitsampling = os.path.join(_sampling,'splitsampling')


if not os.path.exists(undersampling):
    os.makedirs(undersampling)
if not os.path.exists(oversampling):
    os.makedirs(oversampling)
if not os.path.exists(combsampling):
    os.makedirs(combsampling)
if not os.path.exists(splitsampling):
    os.makedirs(splitsampling)

#Create dataframe of WCC Survey data
WCC_Survey = sm.csv_to_pd("WCC_mob_final_no_loc", _preprocesseddir)
print(WCC_Survey)


#UnderSampling
USamp = sm.Sampling(WCC_Survey, _sampling)
USamp.normalise()

#USamp.impute(name = 'Bayseian Ridge', estimators = BayesianRidge())
#USamp.impute(name = 'Decision-Tree Regressor', estimators = DecisionTreeRegressor())
#USamp.impute(name = 'Extra-Trees Regressor', estimators = ExtraTreesRegressor())
#USamp.impute(name = 'K-Neighbours Regressor', estimators
#= KNeighborsRegressor())


USamp.impute()
USamp.train_test_split(0.3)


USamp.near_miss(ratio = 0.7, dataOut = 'Near Miss - Ratio 0.7')






USamp.logistic_regression()
USamp.test_model()

USamp.save_dataset('undersampling')
USamp.save_results('undersampling')
"""
#USamp.save_models()
USamp.PCA_plot()

USamp.Scatter_Plot(external_data = os.path.join(undersampling, "NM_CC.xlsx"))
#OverSampling

OSamp = sm.Sampling(WCC_Survey, oversampling)
OSamp.normalise()
OSamp.impute()
OSamp.train_test_split(0.3)

OSamp.SMOTE(ratio = 0.1, dataOut='0.1_1')
OSamp.SMOTE(ratio = 0.1, dataOut='0.1_2')
OSamp.SMOTE(ratio = 0.1, dataOut='0.1_3')
OSamp.SMOTE(ratio = 0.2, dataOut='0.2_1')
OSamp.SMOTE(ratio = 0.2, dataOut='0.2_2')
OSamp.SMOTE(ratio = 0.2, dataOut='0.2_3')
OSamp.SMOTE(ratio = 0.3, dataOut='0.3_1')
OSamp.SMOTE(ratio = 0.3, dataOut='0.3_2')
OSamp.SMOTE(ratio = 0.3, dataOut='0.3_3')
OSamp.SMOTE(ratio = 0.4, dataOut='0.4_1')
OSamp.SMOTE(ratio = 0.4, dataOut='0.4_2')
OSamp.SMOTE(ratio = 0.4, dataOut='0.4_3')
OSamp.SMOTE(ratio = 0.5, dataOut='0.5_1')
OSamp.SMOTE(ratio = 0.5, dataOut='0.5_2')
OSamp.SMOTE(ratio = 0.5, dataOut='0.5_3')
OSamp.SMOTE(ratio = 0.6, dataOut='0.6_1')
OSamp.SMOTE(ratio = 0.6, dataOut='0.6_2')
OSamp.SMOTE(ratio = 0.6, dataOut='0.6_3')
OSamp.SMOTE(ratio = 0.7, dataOut='0.7_1')
OSamp.SMOTE(ratio = 0.7, dataOut='0.7_2')
OSamp.SMOTE(ratio = 0.7, dataOut='0.7_3')
OSamp.SMOTE(ratio = 0.8, dataOut='0.8_1')
OSamp.SMOTE(ratio = 0.8, dataOut='0.8_2')
OSamp.SMOTE(ratio = 0.8, dataOut='0.8_3')
OSamp.SMOTE(ratio = 0.9, dataOut='0.9_1')
OSamp.SMOTE(ratio = 0.9, dataOut='0.9_2')
OSamp.SMOTE(ratio = 0.9, dataOut='0.9_3')

OSamp.logistic_regression()
OSamp.test_model()
OSamp.Scatter_Plot(external_data=os.path.join(oversampling,'model_metrics_average.xlsx'))
#OSamp.save_dataset('oversampling')
OSamp.save_results('oversampling')

#OSamp.save_models()

#OSamp.PCA_plot()

#Over and Under Sampling
CSamp = sm.Sampling(WCC_Survey, combsampling)

CSamp.normalise()
CSamp.impute()
CSamp.train_test_split(0.3)

CSamp.SMOTE(ratio = 0.4, dataOut='SMOTE')
CSamp.neigbourhood_cleaning(dataIn='SMOTE', dataOut='1')
CSamp.near_miss(ratio = 0.7, dataIn = 'SMOTE', dataOut = '2')
CSamp.near_miss(ratio = 0.8, dataIn = 'SMOTE', dataOut = '3')
CSamp.near_miss(ratio = 0.9, dataIn = 'SMOTE', dataOut = '4')
CSamp.near_miss(ratio = 1.0, dataIn = 'SMOTE', dataOut = '5')
CSamp.cluster_centroids(ratio = 0.8, dataIn = 'SMOTE', dataOut = '6')
CSamp.cluster_centroids(ratio = 0.9, dataIn = 'SMOTE', dataOut = '7')
CSamp.cluster_centroids(ratio = 1.0, dataIn = 'SMOTE', dataOut = '8')


CSamp.logistic_regression()
CSamp.test_model()

CSamp.Bar_Plot(external_data=os.path.join(combsampling,'mean_models.xlsx'), data = 'True')

CSamp.save_dataset('combsampling')
CSamp.save_results('combsampling')
#CSamp.save_models()
CSamp.PCA_plot()

"""


