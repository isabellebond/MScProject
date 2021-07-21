import pandas as pd
import os
import numpy as np
import dataFuncs

#Initialise repositories
_projroot = os.path.abspath('.')
_datadir = os.path.join(_projroot, 'data')
_preprocesseddir = os.path.join(_datadir, 'preprocesseddata')
_rawdir = os.path.join(_datadir, 'rawdata')

#Create dataframe of WCC Survey data
WCC_Survey = dataFuncs.excel_to_pd("WCC2020Num", _preprocesseddir)

data = dataFuncs.RawData(WCC_Survey)
print(data, data.dataframe.head())
#Categorise each column and create dictionary for each value
data.convert_to_int(["S1"], "Gender, S1")
data.convert_to_bool(["Q8a","Q8b","Q8c","Q8d","Q8e","Q8f"], "Council Services, Q8")
data.convert_to_bool(["Q12a","Q12b","Q12c","Q12d","Q12e","Q12f","Q12g","Q12h","Q12i","Q12j","Q12k","Q12l"], "Publications, Q12")
data.convert_to_bool(["Q14a","Q14b","Q14c","Q14d","Q14e","Q14f","Q14g","Q14h","Q14i","Q14j","Q14k","Q14l","Q14m"],"Resource Access, Q14")
data.convert_to_bool(["Q26a","Q26b","Q26c","Q26d","Q26e","Q26f","Q26g","Q26h","Q26i","Q26j","Q26k"], "Financial Worry, Q26")
data.convert_to_bool(["Q29"], "Carer", YesNo = True)

data.dataframe = data.dataframe.drop(["Q27a","Q28a","Q37b","Q37c","Q37d","Q37e","CategoryType","WellbeingType","AcornTypeCode","LowerSuperOutput","weight"], axis = 1)
data.binarize_target_WCC_mobile("WCC_Mobile")
#data.binarize_target_WCC_total("WCC_Total")

data.dataframe.to_csv(os.path.join(_preprocesseddir,"WCCSurvey_mobile_comb.csv"))