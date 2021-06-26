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
Ofcom = dataFuncs.binarize_target_Ofcom(Ofcom)

#-------------------------------------------------------
#-------Population by Age and Gender Data---------------
#-------------------------------------------------------

AgeGender = dataFuncs.excel_to_pd("AgeGender", _preprocesseddir, Skiprows = 7)
#Remove large building data
AgeGender = dataFuncs.delete_flag_removal(AgeGender)
#Bin age data
AgeGender = dataFuncs.column_combine(AgeGender,["Females aged 0-4", "Females aged 10-14", "Females aged 15","Females aged 16-17","Females aged 18-19","Females aged 20-24"], "Females aged under 25" )
AgeGender = dataFuncs.column_combine(AgeGender,["Females aged 25-29","Females aged 30-34","Females aged 35-39","Females aged 40-44"], "Females aged 25-44")
AgeGender = dataFuncs.column_combine(AgeGender,["Females aged 45-49","Females aged 50-54","Females aged 55-59","Females aged 60-64"], "Females aged 45-65")
AgeGender = dataFuncs.column_combine(AgeGender,["Males aged 0-4", "Males aged 10-14", "Males aged 15","Males aged 16-17","Males aged 18-19","Males aged 20-24"],"Males aged under 25")
AgeGender = dataFuncs.column_combine(AgeGender,["Males aged 25-29","Males aged 30-34","Males aged 35-39","Males aged 40-44"],"Males aged 25-44")
AgeGender = dataFuncs.column_combine(AgeGender,["Males aged 45-49","Males aged 50-54","Males aged 55-59","Males aged 60-64"], "Males aged 45-65")

#-------------------------------------------------------
#-------------------Paycheck data-----------------------
#-------------------------------------------------------

Paycheck = dataFuncs.excel_to_pd("Paycheck", _preprocesseddir, Skiprows = 10)
#Remove large building data
Paycheck = dataFuncs.delete_flag_removal(Paycheck)
#Keep only overall markers of income
Paycheck = Paycheck[["Postcode","Mean Income","Median Income","Mode Income","Lower Quartile"]]
#Delete postcodes with no data
Paycheck = Paycheck.dropna()


#-------------------------------------------------------
#----------------Combination data-----------------------
#-------------------------------------------------------
Complete = dataFuncs.dataset_combine(AgeGender,Ofcom,"Postcode")
Complete = dataFuncs.dataset_combine(Complete,Paycheck,"Postcode")
#Output to csv
Complete.to_csv(os.path.join(_preprocesseddir,"Ofcom_binary.csv"))



