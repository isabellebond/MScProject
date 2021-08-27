import pandas as pd
import os
import numpy as np
import json
import pgeocode
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from pandas._libs.tslibs.timedeltas import ints_to_pytimedelta


#Initialise repositories
_projroot = os.path.abspath('.')
_datadir = os.path.join(_projroot, 'data')
_preprocesseddir = os.path.join(_datadir, 'preprocesseddata')
_rawdir = os.path.join(_datadir, 'rawdata')
       
def excel_to_pd(filename, directory, sheetnumber = 0, headernumber = 0, Skiprows = None, indexcol = None):
    """
    Create a pandas dataframe of specified excel spreadsheet

        Parameters:
                filename (string)
                sheetnumber (int): sheetnumber required, default = 0
                headernumber (int): number of header rows in speadsheet, default = 1
        
        Return:
                dataframe: dataframe of input spreadsheet
    """
    dataframe = pd.read_excel(os.path.join(directory, "%s.xlsx"%filename), sheet_name = sheetnumber, header = headernumber ,index_col= indexcol, skiprows = Skiprows)
    return dataframe

def csv_to_pd(filename, directory, headernumber = 0):

    """
    Create a pandas dataframe of specified csv file

        Parameters:
                filename (string)
                headernumber (int): number of header rows in speadsheet, default = 1
        
        Return:
                dataframe: dataframe of input csv

    """
    dataframe = pd.read_csv(os.path.join(directory, "%s.csv"%filename), header = headernumber)
    return dataframe

class RawData:

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.reference = {}
        self.imputed = {}
        self.params = {}

    def convert_to_bool(self, columnname, title, YesNo = False):

        """
        Converts all values in a column to boolean bit values 

            Parameters:
                dataframe
                columnname (list): name of columns in dataframe to convert
                YesNo (bool): True if column values are in Yes/No form
        
            Return:
                dataframe: dataframe with updated column values
        """

        if YesNo:
            for i in columnname:
                column = np.array([])
                for j in self.dataframe.loc[:, i]:
                    if j == "No":
                        column = np.append(column, 0)
                    elif j == "Yes":
                        column = np.append(column, 1)
                    else:
                        column = np.append(column, np.nan)
                self.dataframe[i] = column
        else:
            for i in columnname:
                column = np.array([])
                for j in self.dataframe.loc[:, i]:
                    if pd.isnull(j):
                        column = np.append(column, 0)
                    else:
                        column = np.append(column, 1)
                self.dataframe[i] = column
                self.reference[title] = "bool"
        return self.dataframe

    def convert_to_int(self, columnname, title, reference = {}):

        """
        Converts all values in a column to integer values.
        Each unique string is assigned an integer value and added to a dictionary.
        Dataframe column is updated with integer value.

            Parameters:
                dataframe
                columnname (list): name of columns in dataframe to convert
                order (dict): If specified interger-string pairs are required, add this to reference
        
            Return:
                dataframe: dataframe with updated column values
                reference (dict): Dictionary with key-value pairs to decode 
        """
        k = 0 #value of integer pointer to dictionary value 

        for i in columnname:
            column = np.array([])
            for j in self.dataframe.loc[:, i]:
                if j in reference:
                    j = reference[j]
                    column = np.append(column, j)
                else:
                    reference[j] = k
                    k += 1
                    j = reference[j]
                    column = np.append(column, j)
    
            self.dataframe[i] = column
            self.reference[title] = reference

        return self.dataframe, self.reference
    
    def convert_to_latlong(self, columname):
        geo = pgeocode.Nominatim('gb')
        self.dataframe[columname] = self.dataframe[columname].apply(lambda x:' '.join([x[0:len(x)-3],x[len(x)-3:]]))
        #print(self.dataframe[columname])
        self.dataframe['latitude'] = geo.query_postal_code(self.dataframe[columname].values)['latitude']
        self.dataframe['longitude'] = geo.query_postal_code(self.dataframe[columname].values)['longitude']
        self.dataframe.drop(columname, axis = 1, inplace = True)
        return

    def one_hot_encode(self, columnname, column_prefix = None, drop_other = False):

        one_hot = pd.get_dummies(self.dataframe[columnname], prefix = column_prefix)
        self.dataframe = self.dataframe.join(one_hot)
        self.dataframe.drop(columnname, axis = 1, inplace = True)
        if drop_other ==  True:
            try:
                self.dataframe.drop('%s_other'%column_prefix, axis = 1, inplace = True)
            except KeyError:
                pass 
        
        return self.dataframe

    def cat_combine(self, columname, old_response, new_response = 'other'):
        """
        Combines multiple categorical responses into one

            Parameters:
                dataframe
                columnname(str): name of column in dataframe to apply function to
                old_response(list or int): 
                    if list: list of strings to combine into new response
                    if int: threshold value, all response with less than int occurances will be combined into new response
                new_response(str): new name for combined response

            Return:
                dataframe: dataframe with updated column values
        """

        if type(old_response) == list:
            self.dataframe[columname] = self.dataframe[columname].apply(lambda x:x if x not in old_response else new_response)
        else: 
            #replace categories with < old_reponse answers with new_response
            replace = []
            counts = self.dataframe[columname].value_counts()
            for index, value in counts.items():
                if value < old_response:
                    replace.append(index)
            self.dataframe[columname] = self.dataframe[columname].apply(lambda x:x if x not in replace else new_response)
        
        return self.dataframe
            

    def set_nan(self, columname, NaN_values):

        for column in columname:
            self.dataframe[column] = self.dataframe[column].apply(lambda x:x if x not in NaN_values else np.nan)
        return self.dataframe
    
    def sum_columns(self, columname, new_columname):
        self.dataframe[columname].fillna(0)
        self.dataframe[new_columname] = self.dataframe[columname].sum(axis = 1)
        self.dataframe.drop(columname, axis = 1, inplace = True)
        return self.dataframe
    
    def mean_columns(self, columname, new_columname):
        self.dataframe[new_columname] = self.dataframe[columname].mean(axis=1, skipna = True)
        self.dataframe.drop(columname, axis = 1, inplace = True)
        return self.dataframe

    def binarize_target_WCC_total(self, csv_name):
        """
        Creates target data for Westminster City Council survey
        Takes answers to Q10 and 14 to work out digital exclusion
        Removes Q10 and 14 columns the creates target column
        1 = digitally excluded
        0 = not digitally excluded

            Parameters:
                dataframe: Westminster city council survey dataframe
        
            Return:
                dataframe: dataframe with updated column values
        """
        digitallyExcluded = np.array([])
        dataframe = self.dataframe

        #If people have no devices to access the internet count as digitally excluded
        for i in dataframe.loc[:, "Q10i"]:
            if pd.isnull(i):
                digitallyExcluded = np.append(digitallyExcluded, 0)
            else:
                digitallyExcluded = np.append(digitallyExcluded, 1)

        #If people answer don't know to what devices, look at use in Q14 to make decision
        onlineInfo = ["Q14d","Q14e","Q14f","Q14g","Q14h"]
        Fourteen = np.zeros([len(dataframe.loc[:, "Q14d"])])
    
        #Create array indicative of whether people choose to get info via online means, value of 1 is yes
        for i in onlineInfo:
            column = np.array([])
            for j in dataframe.loc[:, i]:
                if pd.isnull(j):
                    column = np.append(column, 0)
                else:
                    column = np.append(column, 1)
            Fourteen = np.sign(Fourteen + column)

        #Convert answers to 10j to bool
        QTenJ = np.array([])
        for j in dataframe.loc[:,"Q10j"]:
            if pd.isnull(j):
                QTenJ = np.append(QTenJ, 0)
            else:
                QTenJ = np.append(QTenJ, 1)
        

    
        QFourteenK = np.array([])
        for j in dataframe.loc[:,"Q10j"]:
            if pd.isnull(j):
                QFourteenK = np.append(QFourteenK, 0)
            else:
                QFourteenK = np.append(QFourteenK, 1)

        #Update digitally excluded values, drop rows if necessary
        for i in range(0,len(digitallyExcluded)):
            if QTenJ[i] == 1:
                if Fourteen[i] == 1:
                    digitallyExcluded[i] = 0
                elif QFourteenK[i] == 1:
                    dataframe = dataframe.drop([i], axis = 0)
                    digitallyExcluded = np.delete(digitallyExcluded,i)
                    print(i)
                else:
                    digitallyExcluded[i] = 1

        #Add new "Target" column to WCC Survey dataframe
        self.dataframe = dataframe.drop(["Q10a","Q10b", "Q10c","Q10d","Q10e","Q10f","Q10g","Q10h","Q10i","Q10j",
                                    "Q14a","Q14b", "Q14c","Q14d","Q14e","Q14f","Q14g","Q14h","Q14i","Q14j","Q14k","Q14l","Q14m"], axis = 1)

        self.dataframe.insert(0,"Target",digitallyExcluded)

        self.dataframe.to_csv(os.path.join(_preprocesseddir,"%s.csv"%csv_name))
        json.dump(self.reference,open(os.path.join(_preprocesseddir,"%s.json"%csv_name),"w"), indent=2)
        print(self.dataframe.head())
        return self.dataframe

    def binarize_target_WCC_mobile(self, csv_name):
        """
        Creates target data for Westminster City Council survey
        Takes answers to Q10 and if only use mobile phone to access the internet, count as digitally excluded
        1 = digitally excluded
        0 = not digitally excluded

            Parameters:
                dataframe: Westminster city council survey dataframe
        
            Return:
                dataframe: dataframe with updated column values
        """
        dataframe = self.convert_to_bool(["Q10a","Q10b", "Q10c","Q10d","Q10e","Q10f","Q10g","Q10h","Q10i","Q10j"],"Q10, Target")
        dataframe = dataframe[dataframe['Q10h'] == 0]
        dataframe = dataframe[dataframe['Q10i'] == 0]
        dataframe = dataframe[dataframe['Q10j'] == 0]
        print(dataframe)
        dataframe_target = dataframe[["Q10a","Q10c","Q10d","Q10f","Q10g"]]
        
        digitallyExcluded = (np.sign(dataframe_target.sum(axis = 1)) +1 ) % 2 
        self.dataframe = dataframe.drop(["Q10a","Q10b", "Q10c","Q10d","Q10e","Q10f","Q10g","Q10h","Q10i","Q10j"], axis = 1)
        self.dataframe.insert(0,"Target",digitallyExcluded)

        self.dataframe.to_csv(os.path.join(_preprocesseddir,"%s.csv"%csv_name))
        json.dump(self.reference,open(os.path.join(_preprocesseddir,"%s.json"%csv_name),"w"), indent=2)

        return self.dataframe

    def binarize_target_Ofcom(self, csv_name):
        """
        Creates target data for Ofcom data set
        = 0 if >80% have internet speed >30MBs-1 or >50% have internet speed > 1GBs-1
        = 1 otherwise
        Removes internet speed columns from dataset and adds target data
        1 = digitally excluded
        0 = not digitally excluded

            Parameters:
                    dataframe: Ofcom dataframe
            
            Return:
                    dataframe: dataframe with updated column values
        """
        dataframe = self.dataframe
        digitallyExcluded = np.array([])

        #If people have no devices to access the internet count as digitally excluded
        for i in dataframe.loc[:, "Perc_Premises_below_30Mbits"]:
            if i > 20:
                digitallyExcluded = np.append(digitallyExcluded, 1)
            else:
                digitallyExcluded = np.append(digitallyExcluded, 0)
        

        for i in range(0, len(digitallyExcluded)):
            if dataframe.loc[i, "Perc_Gigabit_availability"] > 50:
                digitallyExcluded[i] = 0
                

        self.dataframe = self.dataframe.drop(["Perc_Premises_below_30Mbits", "Perc_Gigabit_availability"], axis = 1)
        print(self.dataframe.head())
        self.dataframe.insert(2,"Target",digitallyExcluded)
        print(self.dataframe.head())

        self.dataframe.to_csv(os.path.join(_preprocesseddir,"%s.csv"%csv_name))
        json.dump(self.reference,open(os.path.join(_preprocesseddir,"%s.json"%csv_name),"w"), indent=2)


        return self.dataframe
    
    def delete_flag_removal(self):
        """
        Removes rows with delete flag set to 1 or with zero population
            Parameters:
                    dataframe
            
            Return:
                    dataframe: dataframe with deleted rows
        """

        for i in range(0,len(self.dataframe.loc[:, "Deleted"])):
            if self.dataframe.loc[i, "Deleted"] == 1:
                self.dataframe = self.dataframe.drop(i, axis = 0)
            try:
                if self.dataframe.loc[i, "Total population"] == 0:
                    self.dataframe = self.dataframe.drop(i, axis = 0)
            except KeyError:
                pass
            
                    
        self.dataframe = self.dataframe.drop("Deleted", axis = 1)

        try:
            self.dataframe = self.dataframe.drop("Total population", axis = 1)
        except KeyError:
            pass
        
        return self.dataframe

    def column_combine(self,columnname,newcolumnname):
        """
        Adds values in specified columns together
        Deletes specified columns
        Creates new column with summed data


            Parameters:
                    dataframe
                    columnname (list): name of columns in dataframe to convert
                    newcolumnname (str): Name of new column to be created
            
            Return:
                    dataframe: dataframe with updated column values
        """
        sum_column = self.dataframe["%s"%columnname[0]]
        dataframe = self.dataframe.drop(columnname[0], axis = 1)

        for i in range(1,len(columnname)):
            sum_column = sum_column + dataframe[columnname[i]]
            dataframe = dataframe.drop(columnname[i], axis = 1)
        
        dataframe[newcolumnname] = sum_column
        self.dataframe = dataframe
        return self.dataframe

    def dataset_combine(self, other, combinecolumn):
        """
        Combines datasets via combinecolumn
        Deletes any whitespace in strings in combine columns
        Deletes rows that are not intersection of both datasets

            Parameters:
                    self: dataframe object to be merged
                    other: dataframe object to be merged
                    combinecolumn (str): Column dataframes to be indexed on
            
            Return:
                    dataframe: merged dataframe
        """
        #Make sure all strings in combine column are in the same format - remove whitespace"
        self.dataframe[combinecolumn] = self.dataframe[combinecolumn].str.replace(" ","")
        other.dataframe[combinecolumn] =  other.dataframe[combinecolumn].str.replace(" ","")

        #Combine dataframes
        self.dataframe = self.dataframe.merge(other.dataframe, how = "inner", on = combinecolumn)
        return self.dataframe

    def save_results(self, path):
        dataframe = pd.DataFrame.from_dict(self.params)    
        dataframe.to_excel(path)
        return

