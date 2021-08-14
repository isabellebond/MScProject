from numpy.core.fromnumeric import amax
import pandas as pd
import os
import numpy as np
import src.tuningFuncs as tf

#Initialise repositories
_projroot = os.path.abspath('.')
_datadir = os.path.join(_projroot, 'data')
_preprocesseddir = os.path.join(_datadir, 'preprocesseddata')
_rawdir = os.path.join(_datadir, 'rawdata')
_src = os.path.join(_projroot,'src')
_WCC_tot = os.path.join(_preprocesseddir,'WCC_tot')
_WTsplitdata = os.path.join(_WCC_tot,'splitdata')
_WCC_mob = os.path.join(_preprocesseddir,'WCC_mob')
_WMsplitdata = os.path.join(_WCC_mob,'splitdata')
_Ofcom = os.path.join(_preprocesseddir,'Ofcom')
_Osplitdata = os.path.join(_Ofcom,'splitdata')

WCC_tot = tf.Tuning(_WTsplitdata)
WCC_tot.read_csv(os.path.join(_WTsplitdata))
print(WCC_tot.dataframe)
WCC_tot.impute()
WCC_tot.elastic_regression()
WCC_tot.save_results(os.path.join(_WCC_tot,'hyper_parameter_test.xlsx'))

WCC_mob = tf.Tuning(_WMsplitdata)
WCC_mob.read_csv(os.path.join(_WMsplitdata))
WCC_mob.impute()
WCC_mob.elastic_regression()
WCC_mob.save_results(os.path.join(_WCC_mob,'hyper_parameter_test.xlsx'))

Ofcom = tf.Tuning(_Osplitdata)
Ofcom.read_csv(os.path.join(_Osplitdata))
Ofcom.impute()
Ofcom.elastic_regression()
Ofcom.save_results(os.path.join(_Ofcom,'hyper_parameter_test.xlsx'))


