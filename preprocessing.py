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

data = dataFuncs.RawData(WCC_Survey)
print(data, data.dataframe.head())
#Categorise each column and create dictionary for each value
data.convert_to_int(["S1"], "Gender, S1")
data.convert_to_int(["Q3"],"Work Status, Q3")
data.convert_to_int(["Q4"], "Ethnicity, Q4")
data.convert_to_int(["Q5a","Q5b","Q5c","Q5d","Q5e","Q5f","Q5g","Q5h","Q5i","Q5j","Q5k","Q5l"], "Concerns, Q5")
data.convert_to_bool(["Q8a","Q8b","Q8c","Q8d","Q8e","Q8f"], "Council Services, Q8")
data.convert_to_int(["Q11a","Q11b","Q11c","Q11d","Q11e"], "ResidentInfo, Q11")
data.convert_to_bool(["Q12a","Q12b","Q12c","Q12d","Q12e","Q12f","Q12g","Q12h","Q12i","Q12j","Q12k","Q12l"], "Publications, Q12")
data.convert_to_int(["Q13a","Q13b","Q13c","Q13d","Q13e","Q13f","Q13g","Q13h","Q13i","Q13j"],"ResourceUtility, Q13")
data.convert_to_bool(["Q14a","Q14b","Q14c","Q14d","Q14e","Q14f","Q14g","Q14h","Q14i","Q14j","Q14k","Q14l","Q14m"],"Resource Access, Q14")
data.convert_to_int(["Q20"],"Background Tolerance, Q20")
data.convert_to_int(["Q21"],"Community Improvement, Q21")
data.convert_to_int(["Q22"],"Own House, Q22")
data.convert_to_int(["Q25"], "Financial Management, Q25")
data.convert_to_bool(["Q26a","Q26b","Q26c","Q26d","Q26e","Q26f","Q26g","Q26h","Q26i","Q26j","Q26k"], "Financial Worry, Q26")
data.convert_to_bool(["Q29"], "Carer", YesNo = True)
data.convert_to_int(["Q30"], "Health, Q30")
data.convert_to_int(["Q32","Q33","Q34","Q35"], "Mental Health, Q32-5")
data.convert_to_int(["Q36a","Q36b","Q36c"],"Lonliness, Q36")
data.convert_to_int(["Q31"], "Healthcare Access, Q31")
data.convert_to_int(["Q37a"], "Nationality, Q37")
data.convert_to_int(["Q38"], "Uk Living, Q38")
data.convert_to_int(["Postcode_clean"], "Postcode")
data.convert_to_int(["Ward"], "Ward")
data.convert_to_int(["OutputArea"], "OutputArea")

data.dataframe = data.dataframe.drop(["Q27a","Q28a","Q37b","Q37c","Q37d","Q37e","CategoryType","WellbeingType","AcornTypeCode","LowerSuperOutput","weight"], axis = 1)
data.binarize_target_WCC_mobile("WCC_Mobile")
data.binarize_target_WCC_total("WCC_Total")

WCC_Survey.to_csv(os.path.join(_preprocesseddir,"WCCSurvey_mobile.csv"))
