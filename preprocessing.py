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
WCC_Survey = dataFuncs.excel_to_pd("WCC2020", _preprocesseddir)

#Categorise each column and create dictionary for each value
WCC_Survey,Q1 = dataFuncs.convert_to_int(WCC_Survey,["S1"])
WCC_Survey,Q2 = dataFuncs.convert_to_int(WCC_Survey,["AGE"])
WCC_Survey,Q3 = dataFuncs.convert_to_int(WCC_Survey,["Q3"])
WCC_Survey,Q4 = dataFuncs.convert_to_int(WCC_Survey,["Q4"])
WCC_Survey,Q5 = dataFuncs.convert_to_int(WCC_Survey,["Q5a","Q5b","Q5c","Q5d","Q5e","Q5f","Q5g","Q5h","Q5i","Q5j","Q5k","Q5l"])
WCC_Survey = dataFuncs.convert_to_bool(WCC_Survey,["Q8a","Q8b","Q8c","Q8d","Q8e","Q8f"])
WCC_Survey,Q11 = dataFuncs.convert_to_int(WCC_Survey,["Q11a","Q11b","Q11c","Q11d","Q11e"])
WCC_Survey = dataFuncs.convert_to_bool(WCC_Survey,["Q12a","Q12b","Q12c","Q12d","Q12e","Q12f","Q12g","Q12h","Q12i","Q12j","Q12k","Q12l"])
WCC_Survey,Q13 = dataFuncs.convert_to_int(WCC_Survey,["Q13a","Q13b","Q13c","Q13d","Q13e","Q13f","Q13g","Q13h","Q13i","Q13j"])
WCC_Survey,Q20 = dataFuncs.convert_to_int(WCC_Survey,["Q20"])
WCC_Survey,Q21 = dataFuncs.convert_to_int(WCC_Survey,["Q21"])
WCC_Survey,Q22 = dataFuncs.convert_to_int(WCC_Survey,["Q22"])
WCC_Survey,Q25 = dataFuncs.convert_to_int(WCC_Survey,["Q25"])
WCC_Survey = dataFuncs.convert_to_bool(WCC_Survey,["Q26a","Q26b","Q26c","Q26d","Q26e","Q26f","Q26g","Q26h","Q26i","Q26j","Q26k"])
WCC_Survey = dataFuncs.convert_to_bool(WCC_Survey,["Q29"], YesNo = True)
WCC_Survey,Q30 = dataFuncs.convert_to_int(WCC_Survey,["Q30"])
WCC_Survey,Q32 = dataFuncs.convert_to_int(WCC_Survey,["Q32","Q33","Q34","Q35"])
WCC_Survey,Q36 = dataFuncs.convert_to_int(WCC_Survey,["Q36a","Q36b","Q36c"])
WCC_Survey,Q31 = dataFuncs.convert_to_int(WCC_Survey,["Q31"])
WCC_Survey,Q37 = dataFuncs.convert_to_int(WCC_Survey,["Q37a"])
WCC_Survey,Q38 = dataFuncs.convert_to_int(WCC_Survey,["Q38"])
WCC_Survey,Postcode = dataFuncs.convert_to_int(WCC_Survey,["Postcode_clean"])
WCC_Survey,Ward = dataFuncs.convert_to_int(WCC_Survey,["Ward"])

WCC_Survey = WCC_Survey.drop(["Q27a","Q28a","Q37b","Q37c","Q37d","Q37e","CategoryType","WellbeingType","AcornTypeCode","OutputArea","LowerSuperOutput","weight"], axis = 1)
WCC_Survey = dataFuncs.binarize_target_WCC_mobile(WCC_Survey)

WCC_Survey.to_csv(os.path.join(_preprocesseddir,"WCCSurvey_mobile.csv"))

