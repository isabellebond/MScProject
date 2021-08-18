import pandas as pd
import os
import numpy as np
import src.plotFuncs as pf
import matplotlib.pyplot as plt

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
  
under = pd.read_excel(os.path.join(undersampling, 'undersampling_model_metrics.xlsx'), sheet_name = 'Transpose', index_col =  'Unnamed: 0')
under_test = under.loc[:, under.columns.str.contains('test')]
plt.figure()
under_test.plot(kind = 'bar',rot = 90)
plt.show()

over = pd.read_excel(os.path.join(oversampling, 'oversampling_model_metrics.xlsx'), sheet_name = 'Transpose', index_col =  'Unnamed: 0')
over_test = over.loc[:, over.columns.str.contains('test')]
plt.figure()
over_test.plot(kind = 'bar', use_index = True, title = 'SMOTE oversampling', rot = 90)
plt.show()

comb = pd.read_excel(os.path.join(combsampling, 'combsampling_model_metrics.xlsx'), sheet_name = 'Transpose', index_col =  'Unnamed: 0')
comb_test = comb.loc[:, comb.columns.str.contains('test')]
plt.figure()
comb_test.plot(kind = 'bar',rot = 90)
plt.show()

