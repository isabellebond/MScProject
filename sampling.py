import pandas as pd
import os
import numpy as np
import src.samplingFuncs as sm

#Initialise repositories
_projroot = os.path.abspath('.')
_datadir = os.path.join(_projroot, 'data')
_preprocesseddir = os.path.join(_datadir, 'preprocesseddata')
_rawdir = os.path.join(_datadir, 'rawdata')
_src = os.path.join(_projroot,'src')

#Create dataframe of WCC Survey data
WCC_Survey = sm.csv_to_pd("WCC_total_final_no_loc", _preprocesseddir)
data = sm.Sampling(WCC_Survey)
data.normalise(inplace=True)
data.train_test_split(0.3)

data.cluster_centroids(save = True, inplace = True)
data.SMOTE(inplace = True)
data.data_split("WCC_tot","WCC_tot")
data.save_dataset("WCC_tot","WCC_tot")

#Create dataframe of WCC Survey data
WCC_Survey = sm.csv_to_pd("WCC_mob_final_no_loc", _preprocesseddir)
data = sm.Sampling(WCC_Survey)
data.normalise(inplace=True)
data.train_test_split(0.3)

data.cluster_centroids(save = True, inplace = True)
data.SMOTE(inplace = True)
data.data_split("WCC_mob","WCC_mob")
data.save_dataset("WCC_mob","WCC_mob")

