#Complete preprocessing of data before sampling

import pandas as pd
import os
import numpy as np
import src.rawdataFuncs as rd

#Initialise repositories
_projroot = os.path.abspath('.')
_datadir = os.path.join(_projroot, 'data')
_preprocesseddir = os.path.join(_datadir, 'preprocesseddata')
_rawdir = os.path.join(_datadir, 'rawdata')

#WCC Survey

WCC_Survey = rd.excel_to_pd("WCC2020Num", _preprocesseddir)

data = rd.RawData(WCC_Survey)
print(data, data.dataframe.head())

#---------------------------------------------------
#Categorise each column and apply relevant functions
#---------------------------------------------------

#Q2 Gender
data.convert_to_int(["S1"], "Gender,S1")

#Q3 Work status
data.cat_combine('Q3', ['Working - Part Time (8 - 16 HRS)', 'Working - Part Time (17 - 29 HRS)'], 'Working - Part Time (8 - 29 HRS)')
data.cat_combine('Q3', 20)
data.one_hot_encode('Q3', column_prefix = 'Q3', drop_other = True)

#Q4 Ethnicity
data.set_nan(['Q4'], ['Refused'])
data.cat_combine('Q4', ['White - British', 'White - Western European (type in)'], 'White - Western European')
data.cat_combine('Q4', 10)
data.one_hot_encode('Q4', column_prefix = 'Q4', drop_other = True)

#Q5 Concerns
Q5 = ["Q5a","Q5b","Q5c","Q5d","Q5e","Q5f","Q5g","Q5h","Q5i","Q5j","Q5k","Q5l"]
data.set_nan(Q5, [6])

#Q8 Council services accessed
#Create a single column with summed values from other columns
Q8 = ["Q8a","Q8b","Q8c","Q8d","Q8e","Q8f"]
data.convert_to_bool(Q8, 'Q8 services')
data.sum_columns(Q8, 'Number of Council Services accessed')

#Q11 information about public services
#Take mean of information across all services
Q11 = ["Q11a","Q11b","Q11c","Q11d","Q11e"]
data.set_nan(Q11, [5,6])
data.mean_columns(Q11, 'Information about public services')

#Q12 what council content is accessed
#Categorise Q12 as access online council content
Q12_online = ["Q12d","Q12e","Q12g","Q12h","Q12j"]
Q12_offline = ["Q12a","Q12b","Q12c","Q12f","Q12i"]
Q12_other = ["Q12k","Q12l"]

data.convert_to_bool(Q12_online,'Q12 Online')
data.convert_to_bool(Q12_offline,'Q12 Offline')
data.sum_columns(Q12_online, 'Q12 Online')
data.sum_columns(Q12_offline, 'Q12 Offline')


data.dataframe.drop(Q12_other, axis =1, inplace = True)

#Q13 usefulness of council content
Q13_online = ["Q13d","Q13e","Q13g","Q13h","Q13j"]
Q13_offline = ["Q13a","Q13b","Q13c","Q13f","Q13i"]

data.mean_columns(Q13_online, 'Usefulness of online publication')
data.mean_columns(Q13_offline, 'Usefulness of offline publication')

#Q21 Community improvement
data.set_nan(['Q21'],[5])

#Q22 house ownership
data.cat_combine('Q22', 20)
data.one_hot_encode('Q22','Q22', drop_other = True)

#Q25 
data.set_nan(['Q25'],[10])

#Q26 Financial worry
Q26_Worry = ["Q26a","Q26b","Q26c","Q26d","Q26e","Q26f","Q26g","Q26h"]
Q26_other = ["Q26i","Q26j","Q26k"]

data.convert_to_bool(Q26_Worry, 'Q26 Worry')
data.sum_columns(Q26_Worry, 'Financial Worry')
data.dataframe.drop(Q26_other, axis = 1, inplace = True)

#Q29, carer
data.convert_to_bool(['Q29'], 'Q29, carer', YesNo = True)

#Q31 GP access
key = {'Private GP in Westminster/Elsewhere':2,
     'NHS GP in the Westminster City Council Area':1,
     'NHS GP Elsewhere in London': 1,
     'NHS GP Outside of London but in the UK':1,
     '(Don\'t know)': np.nan,
     'None of these': 0
     }
data.convert_to_int(['Q31'], 'Q31 - GP', key)

#Q36 loneliness
data.set_nan(['Q36a','Q36b','Q36c'], [4])

#Q37, UK national
data.convert_to_bool(['Q37a'], "Q37, UK national")
data.dataframe.drop(['Q37b', 'Q37c','Q37d','Q37e'], axis = 1, inplace = True)

#Q38
data.set_nan(['Q38'], [10])

#Ward
#data.convert_to_int(['Ward'], 'Ward')

#Drop near empty columns (analysis from jupyter notebook)
data.dataframe.drop(['Q27a','Q28a','Q5f','Q5k','Usefulness of online publication'], axis = 1, inplace = True)

#Drop columns for location
#data.convert_to_latlong('Postcode_clean')

#print(data.dataframe['latitude'])
data.dataframe.drop(['Postcode_clean','Ward','CategoryType','WellbeingType','LowerSuperOutput','weight'], axis = 1, inplace = True)

data.binarize_target_WCC_total('WCC_total_final_no_loc')
#data.binarize_target_WCC_mobile('WCC_mob_final_no_loc')


