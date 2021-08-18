import pandas as pd
import os
import numpy as np
import src.samplingFuncs as sm

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
if not os.path.exists(undersampling):
    os.makedirs(undersampling)
if not os.path.exists(oversampling):
    os.makedirs(oversampling)
if not os.path.exists(combsampling):
    os.makedirs(combsampling)

#Create dataframe of WCC Survey data
WCC_Survey = sm.csv_to_pd("WCC_total_final", _preprocesseddir)
print(WCC_Survey)
#UnderSampling
USamp = sm.Sampling(WCC_Survey, undersampling)
USamp.normalise()
USamp.impute()
USamp.train_test_split(0.3)


USamp.tomek_links()
USamp.near_miss()
USamp.neigbourhood_cleaning()
USamp.cluster_centroids(ratio = 0.125, dataOut = 'ClusterCentroid_8')
USamp.cluster_centroids(ratio = 0.25, dataOut = 'ClusterCentroid_4')
USamp.cluster_centroids(ratio = 0.5, dataOut = 'ClusterCentroid_2')

USamp.logistic_regression()
USamp.test_model()

USamp.save_dataset('undersampling')
USamp.save_results('undersampling')
USamp.save_models()

#OverSampling
OSamp = sm.Sampling(WCC_Survey, oversampling)
OSamp.normalise()
OSamp.impute()
OSamp.train_test_split(0.3)

OSamp.SMOTE(ratio = 0.1, dataOut='SMOTE_0.1')
OSamp.SMOTE(ratio = 0.1, dataOut='SMOTE_0.2')
OSamp.SMOTE(ratio = 0.1, dataOut='SMOTE_0.3')
OSamp.SMOTE(ratio = 0.1, dataOut='SMOTE_0.4')
OSamp.SMOTE(ratio = 0.1, dataOut='SMOTE_0.5')
OSamp.SMOTE(ratio = 0.1, dataOut='SMOTE_0.6')
OSamp.SMOTE(ratio = 0.1, dataOut='SMOTE_0.7')
OSamp.SMOTE(ratio = 0.1, dataOut='SMOTE_0.8')
OSamp.SMOTE(ratio = 0.1, dataOut='SMOTE_0.9')

OSamp.logistic_regression()
OSamp.test_model()

OSamp.save_dataset('oversampling')
OSamp.save_results('oversampling')
OSamp.save_models()

#Over and Under Sampling
CSamp = sm.Sampling(WCC_Survey, combsampling)
CSamp.normalise()
CSamp.impute()
CSamp.train_test_split(0.3)


CSamp.tomek_links()
CSamp.near_miss()
CSamp.neigbourhood_cleaning()
CSamp.cluster_centroids(ratio = 0.125, dataOut = 'ClusterCentroid_8')
CSamp.cluster_centroids(ratio = 0.25, dataOut = 'ClusterCentroid_4')
CSamp.cluster_centroids(ratio = 0.5, dataOut = 'ClusterCentroid_2')

CSamp.SMOTE(ratio = 1, dataOut='Smote_Tomek_1', dataIn = 'Tomeklinks')
CSamp.SMOTE(ratio = 0.5, dataOut='Smote_Tomek_2', dataIn = 'Tomeklinks')
CSamp.SMOTE(ratio = 0.25, dataOut='Smote_Tomek_4', dataIn = 'Tomeklinks')
CSamp.SMOTE(ratio = 0.125, dataOut='Smote_Tomek_8', dataIn = 'Tomeklinks')

CSamp.SMOTE(ratio = 1, dataOut='Smote_NC_1', dataIn = 'Neighbourhood Cleaning')
CSamp.SMOTE(ratio = 0.5, dataOut='Smote_NC_2', dataIn = 'Neighbourhood Cleaning')
CSamp.SMOTE(ratio = 0.25, dataOut='Smote_NC_4', dataIn = 'Neighbourhood Cleaning')
CSamp.SMOTE(ratio = 0.125, dataOut='Smote_NC_8', dataIn = 'Neighbourhood Cleaning')

#CSamp.SMOTE(ratio = 1, dataOut='Smote_NM_1', dataIn = 'NearMiss')
#CSamp.SMOTE(ratio = 0.5, dataOut='Smote_NM_2', dataIn = 'NearMiss')
#CSamp.SMOTE(ratio = 0.125, dataOut='Smote_NM_8', dataIn = 'NearMiss')
#CSamp.SMOTE(ratio = 0.25, dataOut='Smote_NM_4', dataIn = 'NearMiss')

CSamp.SMOTE(ratio = 1, dataOut='Smote_CC_1_8', dataIn = 'ClusterCentroid_8')
CSamp.SMOTE(ratio = 0.5, dataOut='Smote_CC_2_8', dataIn = 'ClusterCentroid_8')
CSamp.SMOTE(ratio = 0.25, dataOut='Smote_CC_4_8', dataIn = 'ClusterCentroid_8')

CSamp.SMOTE(ratio = 0.5, dataOut='Smote_CC_2_8', dataIn = 'ClusterCentroid_4')


CSamp.save_dataset('combsampling')
CSamp.save_results('combsampling')

CSamp.logistic_regression()
CSamp.test_model()

CSamp.save_models()

#Splitting Dataset






