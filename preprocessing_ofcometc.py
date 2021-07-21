import pandas as pd
import os
import numpy as np
import dataFuncs

#Initialise repositories
_projroot = os.path.abspath('.')
_datadir = os.path.join(_projroot, 'data')
_preprocesseddir = os.path.join(_datadir, 'preprocesseddata')
_rawdir = os.path.join(_datadir, 'rawdata')

#-------------------------------------------------------
#---------------Ofcom Data------------------------------
#-------------------------------------------------------

Ofcom = dataFuncs.excel_to_pd("Ofcom", _preprocesseddir)
#create Target dataset
ofcom_data = dataFuncs.RawData(Ofcom)



#-------------------------------------------------------
#-------Population by Age and Gender Data---------------
#-------------------------------------------------------

AgeGender = dataFuncs.excel_to_pd("AgeGender", _preprocesseddir, Skiprows = 7)
age_data = dataFuncs.RawData(AgeGender)
#Remove large building data
age_data.delete_flag_removal()
#Bin age data
age_data.column_combine(["Females aged 0-4", "Females aged 10-14", "Females aged 15","Females aged 16-17","Females aged 18-19","Females aged 20-24"], "Females aged under 25" )
age_data.column_combine(["Females aged 25-29","Females aged 30-34","Females aged 35-39","Females aged 40-44"], "Females aged 25-44")
age_data.column_combine(["Females aged 45-49","Females aged 50-54","Females aged 55-59","Females aged 60-64"], "Females aged 45-65")
age_data.column_combine(["Males aged 0-4", "Males aged 10-14", "Males aged 15","Males aged 16-17","Males aged 18-19","Males aged 20-24"],"Males aged under 25")
age_data.column_combine(["Males aged 25-29","Males aged 30-34","Males aged 35-39","Males aged 40-44"],"Males aged 25-44")
age_data.column_combine(["Males aged 45-49","Males aged 50-54","Males aged 55-59","Males aged 60-64"], "Males aged 45-65")

#-------------------------------------------------------
#-------------------Paycheck data-----------------------
#-------------------------------------------------------

Paycheck = dataFuncs.excel_to_pd("Paycheck", _preprocesseddir, Skiprows = 10)
#Remove large building data
paycheck_data = dataFuncs.RawData(Paycheck)
paycheck_data.delete_flag_removal()
#Keep only overall markers of income
paycheck_data.dataframe = paycheck_data.dataframe[["Postcode","Mean Income","Median Income","Mode Income","Lower Quartile"]]
#Delete postcodes with no data
paycheck_data.dataframe.dropna(inplace = True)


#-------------------------------------------------------
#----------------Combination data-----------------------
#-------------------------------------------------------
ofcom_data.dataset_combine(age_data,"Postcode")
ofcom_data.dataset_combine(paycheck_data,"Postcode")

#Postcode and postcode area to int
ofcom_data.convert_to_int(["Postcode", "PC_Area"])

#Output to csv
ofcom_data.binarize_target_Ofcom("Ofcom_Combine")
