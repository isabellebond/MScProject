import pandas as pd
import os
import numpy as np
import json
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC

import pylab as pl

#Initialise repositories
_projroot = os.path.abspath('.')
_datadir = os.path.join(_projroot, 'data')
_preprocesseddir = os.path.join(_datadir, 'preprocesseddata')
_rawdir = os.path.join(_datadir, 'rawdata')
_imagesdir = os.path.join(_projroot,'images')
_overunderdir = os.path.join(_imagesdir,'overunder')

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

    try:
        dataframe.drop("Unnamed: 0", axis = 1, inplace = True)
    except(KeyError):
        pass

    return dataframe

class Sampling():
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.reference = {}
        self.target = self.dataframe['Target']
        self.features = self.dataframe.drop('Target', axis = 1)
        self.X_mean = None
    
    def normalise(self, inplace = False):
        """
        Create new dataframe with all continuous features normalised.

            Parameters:
                dataframe
        
            Return:
                normdata: dataframe with normalised column values
        """
        cont = list(set(list(self.dataframe.select_dtypes(exclude=['object']).columns))-set(['Target']))
        self.normalised = self.dataframe[cont]/self.dataframe[cont].max()
        
        if inplace == True:
            self.features[cont] = self.normalised
            print(self.features[cont])
        return self.normalised
    
    def list_cat_features(self):
        """
        Create mask of categorical features in feature vector

            Parameters:
                self
        
            Return:
                self.contfeatures(list): List of len self.features, true if variable is catgorical
        """

        mask = self.X_train.dtypes
        mask[(mask == 'object')] = True
        mask[(mask != True)] = False

        self.catfeatures = mask

        return self.catfeatures

    def train_test_split(self, testsize, randomstate = None):
        """Splits self.dataframe using sklearn.train_test_split
        Creates new class attributes X_train, y_train, X_test, y_test
            Parameters:
                dataframe
                testsize(float): between 0 and 1. Proportion of dataset used for testing
                randomstate(int): set to specific int for repeatable splitting of data
        
            Return:
                self.X_train(dataframe): feature vectors for training data
                self.X_test(dataframe): feature vectors for test data
                self.y_train(dataframe): target for training data
                self.y_test(dataframe): target vectors for test data
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features, self.target, test_size=testsize, random_state=randomstate)
        
        return self.X_train, self.X_test, self.y_train, self.y_test

    def data_split(self, filename = None, foldername = None):
        """
        Divides total features in majority class into four smaller sub datasets.
        Appends all features in minority class into new subsets of data
        Saves new datasets as CSV files 
        Outputs first 2 principle components onto graph and saves in images

            Parameters:
                dataframe
                filename(str): filename of output files (save in preprocesseddir)
        
            Return:
        """
        class1 = []
        class2 = []
        for i in range(0,len(self.y_train.ravel())):
            if self.y_train.ravel()[i] == 0:
                class1.append(i)
            else:
                class2.append(i)

        print(type(self.X_train))

        X_class1 = self.X_train.iloc[class1,:]
        X_class2 = self.X_train.iloc[class2,:]
        y_class2 = np.ones((len(class2),1))

            
        Xa,Xb = train_test_split(X_class1, test_size = 0.5)
        X1, X2 = train_test_split(Xa, test_size = 0.5)
        X3, X4 = train_test_split(Xb, test_size = 0.5)

        X = [X1,X2,X3,X4]
        k = 1

        path = os.path.join(_preprocesseddir,foldername)
        path2 = os.path.join(path,"splitdata")
        if not os.path.exists(path):
            os.mkdir(path)
        if not os.path.exists(path2):
            os.mkdir(path2)
        for item in X:
            y_class1 = np.zeros((len(item),1))
            y_tot = np.append(y_class1, y_class2, axis = 0)
            X_new = np.append(item, X_class2, axis = 0)
            data_tot = np.append(y_tot, X_new, axis = 1)
            df = pd.DataFrame(data = data_tot, columns = self.features.columns.insert(0,"Target"))
            print(self.X_mean.keys())
            for column in self.X_mean.keys():
                print(column)
                df[column] = df[column].apply( lambda x: np.nan if x == self.X_mean.get(column) else x)

            df.to_csv(os.path.join(path2,"%s-%s.csv"%(filename,k)))
            #self.PCA_plot("%s-%s"%(filename,k),"%s-%s"%(filename,k))
            k+=1

        return

    def cat_to_int(self, columnname, title, catorder = None):
        """
        Converts all values in a column to integer values.
        Each vale that contains a string in 'catorder' is assigned an integer value and added to a dictionary.
        Dataframe column is updated with integer value.

            Parameters:
                dataframe
                columnname (list): name of columns in dataframe to convert
                title (str): name of dictionary item in self.reference
                catoorder (list): strings in order of expected numbering
        
            Return:
                dataframe: dataframe with updated column values
                reference (dict): Dictionary with key-value pairs to decode 
        """
        reference = {}
        k = 0 #value of interger pointer to dictionary value 

        if catorder == None:
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
        
        else:
            for i in catorder:
                reference[i] = k
                k += 1
            self.reference[title] = reference

            for i in columnname:
                column = np.array([])
                for j in self.dataframe.loc[:, i]:
                    for key, value in reference.items():
                        if key in j:
                            column= np.append(column, value)
                
                self.dataframe[i] = column
        
        return self.dataframe, self.reference
                
    def PCA_plot(self, X, y, figtitle, figname, inplace = False):
        # Use principal component to condense the 10 features to 2 features
        pca = PCA(n_components=2).fit(X)
        pca_2d = pca.transform(X)

        for i in range(0, pca_2d.shape[0]):
            if y[i] == 0:
                c1 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r', marker='+')
            elif y[i] == 1:
                c2 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='b', marker='x')  
    
        pl.legend([c1, c2], ['Not Digitally Excluded', 'Digitally Exluded'])
        pl.xlabel('Principle Compenent 1')
        pl.ylabel('Principle Compenent 2')
        pl.title(figtitle)
        pl.savefig(os.path.join(_overunderdir, figname), dpi = 1000)
        pl.show()

        return

    def tomek_links(self, figname = 'Tomeklinks', inplace = False, save = False):
        """Undersample data according to Tomek Links algorithm
        Plots first two principle components of sampled data
        Returns resampled feature vector and training data
        Resampling according to continuous data only

        Parameters:
                dataframe
                figname (str): name of output figure
                inplace (bool): true if want to replace self.X_train and self.y_train with resampled data
                save (bool): true if want to save resampled data in csv files
        
            Return:
                X_resample(dataframe): resampled feature data
                y_resample(dataframe): resampled target data
        """

        mask = self.list_cat_features()
        if self.X_mean == None:
            self.X_mean = self.X_train.mean()

        contfeatures = self.X_train.loc[:,mask==False].fillna(self.X_train.mean())

        sampler = TomekLinks() 

        X_resample, y_resample = sampler.fit_resample(contfeatures, self.y_train)

        print('TomekLinks undersampling {}'.format(Counter(y_resample)))
        self.PCA_plot(X_resample, y_resample,figname, figname)  
        
        X_resample = X_resample.merge(self.X_train.fillna(self.X_train.mean()), how = "inner", on = list(X_resample.columns))

        if save == True:
            X_resample.to_csv(os.path.join(_preprocesseddir,"%s-features.csv"%figname))
            y_resample.to_csv(os.path.join(_preprocesseddir,"%s-target.csv"%figname))
        if inplace == True:
            self.X_train = X_resample
            self.y_train = y_resample

        return X_resample, y_resample

    def near_miss(self, figname = 'NearMiss', inplace = False, save = False):
        """Undersample data according to Near Miss algorithm
        Plots first two principle components of sampled data
        Returns resampled feature vector and training data
        Resampling according to continuous data only

        Parameters:
                dataframe
                figname (str): name of output figure
                inplace (bool): true if want to replace self.X_train and self.y_train with resampled data
                save (bool): true if want to save resampled data in csv files
        
            Return:
                X_resample(dataframe): resampled feature data
                y_resample(dataframe): resampled target data
        """
        mask = self.list_cat_features()
        if self.X_mean == None:
            self.X_mean = self.X_train.mean()

        contfeatures = self.X_train.loc[:,mask==False].fillna(self.X_train.mean())

        sampler = NearMiss() 

        X_resample, y_resample = sampler.fit_resample(contfeatures, self.y_train)
        print('NearMiss undersampling {}'.format(Counter(y_resample)))
        self.PCA_plot(X_resample, y_resample,figname, figname)  
        
        X_resample = X_resample.merge(self.X_train.fillna(self.X_train.mean()), how = "inner", on = list(X_resample.columns))

        if save == True:
            X_resample.to_csv(os.path.join(_preprocesseddir,"%s-features.csv"%figname))
            y_resample.to_csv(os.path.join(_preprocesseddir,"%s-target.csv"%figname))
        
        if inplace == True:
            self.X_train = X_resample
            self.y_train = y_resample
        
        return X_resample, y_resample

    def cluster_centroids(self, ratio = 0.125, figname = 'ClusterCentroid', save = False, inplace = False):
        """Undersample data according to Cluster centroid algorithm
        Plots first two principle components of sampled data
        Returns resampled feature vector and training data
        Resampling according to continuous data only

        Parameters:
                dataframe
                ratio(float): expected ratio of minority class/majority class after sampling
                figname (str): name of output figure
                inplace (bool): true if want to replace self.X_train and self.y_train with resampled data
                save (bool): true if want to save resampled data in csv files
        
            Return:
                X_resample(dataframe): resampled feature data
                y_resample(dataframe): resampled target data
        """

        mask = self.list_cat_features()
        if self.X_mean == None:
            self.X_mean = self.X_train.mean()
        self.X_train.fillna(self.X_mean, inplace = True)
        self.X_train.to_csv('Test.csv')
        sampler = ClusterCentroids(sampling_strategy = ratio, voting = "hard") 
        print(self.X_train.isnull().values.any())
        X_resample, y_resample = sampler.fit_resample(self.X_train, self.y_train)
        print('Cluster Centroid undersampling {}'.format(Counter(y_resample)))
        print(X_resample.shape,y_resample.shape)
        #self.PCA_plot(X_resample, y_resample,figname, figname)  
        
        X_resample = X_resample.merge(self.X_train, how = "inner", on = list(X_resample.columns))
        print(X_resample.shape,y_resample.shape)
        if save == True:
            X_resample.to_csv(os.path.join(_preprocesseddir,"%s-features.csv"%figname))
            y_resample.to_csv(os.path.join(_preprocesseddir,"%s-target.csv"%figname))
        
        if inplace == True:
            self.X_train = X_resample
            self.y_train = y_resample

        return X_resample, y_resample

    def neigbourhood_cleaning(self, figname = 'Neighbourhood Cleaning', save = False, inplace = False):
        """Undersample data according to Near Miss algorithm
        Plots first two principle components of sampled data
        Returns resampled feature vector and training data
        Resampling according to continuous data only

        Parameters:
                dataframe
                figname (str): name of output figure
                inplace (bool): true if want to replace self.X_train and self.y_train with resampled data
                save (bool): true if want to save resampled data in csv files
        
            Return:
                X_resample(dataframe): resampled feature data
                y_resample(dataframe): resampled target data
        """
        mask = self.list_cat_features()
        if self.X_mean == None:
            self.X_mean = self.X_train.mean()

        contfeatures = self.X_train.loc[:,mask==False].fillna(self.X_train.mean())

        sampler = ClusterCentroids() 

        X_resample, y_resample = sampler.fit_resample(contfeatures, self.y_train)
        print('Neighbourhood Cleaning undersampling {}'.format(Counter(y_resample)))
        self.PCA_plot(X_resample, y_resample,figname, figname)  
        
        X_resample = X_resample.merge(self.X_train.fillna(self.X_train.mean()), how = "inner", on = list(X_resample.columns))

        if save == True:
            X_resample.to_csv(os.path.join(_preprocesseddir,"%s-features.csv"%figname))
            y_resample.to_csv(os.path.join(_preprocesseddir,"%s-target.csv"%figname))
        
        if inplace == True:
            self.X_train = X_resample
            self.y_train = y_resample

        return X_resample, y_resample
    
    def SMOTE(self, ratio = 0.25, k = 5, figname = "SMOTE", inplace = False, save = False):
        """Oversample minority data according to SMOTE algorithm
        Plots first two principle components of sampled data
        Returns resampled feature vector and training data
        Resampling includes categorical features

        Parameters:
                dataframe
                ratio(float): expected ratio of minority class/majority class after sampling
                figname (str): name of output figure
                inplace (bool): true if want to replace self.X_train and self.y_train with resampled data
                save (bool): true if want to save resampled data in csv files
        
            Return:
                X_resample(dataframe): resampled feature data
                y_resample(dataframe): resampled target data
        """
        
        mask = self.list_cat_features()
        #if not self.X_mean:
        #    self.X_mean = self.X_train.mean()
        self.X_train.loc[:,mask==True] = self.X_train.loc[:,mask==True].astype(str)
        print(self.X_train.dtypes)
        mask = list(self.list_cat_features())
        sampler = SMOTE( sampling_strategy = ratio, k_neighbors = k) 
        self.X_train = self.X_train.fillna(self.X_train.mean())
        X_resample, y_resample = sampler.fit_resample(self.X_train, self.y_train)

        if save == True:
            X_resample.to_csv(os.path.join(_preprocesseddir,"%s-features.csv"%figname))
            y_resample.to_csv(os.path.join(_preprocesseddir,"%s-target.csv"%figname))

        if inplace == True:
            self.X_train = X_resample
            self.y_train = y_resample

        return X_resample, y_resample

    def SMOTE_cat(self, ratio = 0.25, k = 5, figname = "SMOTE", inplace = False, save = False):
        """Oversample minority data according to SMOTE algorithm
        Plots first two principle components of sampled data
        Returns resampled feature vector and training data
        Resampling includes categorical features

        Parameters:
                dataframe
                ratio(float): expected ratio of minority class/majority class after sampling
                figname (str): name of output figure
                inplace (bool): true if want to replace self.X_train and self.y_train with resampled data
                save (bool): true if want to save resampled data in csv files
        
            Return:
                X_resample(dataframe): resampled feature data
                y_resample(dataframe): resampled target data
        """
        
        mask = self.list_cat_features()
        #if not self.X_mean:
        #    self.X_mean = self.X_train.mean()
        self.X_train.loc[:,mask==True] = self.X_train.loc[:,mask==True].astype(str)
        print(self.X_train.dtypes)
        mask = list(self.list_cat_features())
        sampler = SMOTENC(mask, sampling_strategy = ratio, k_neighbors = k) 
        self.X_train = self.X_train.fillna(self.X_train.mean())
        X_resample, y_resample = sampler.fit_resample(self.X_train, self.y_train)

        if save == True:
            X_resample.to_csv(os.path.join(_preprocesseddir,"%s-features.csv"%figname))
            y_resample.to_csv(os.path.join(_preprocesseddir,"%s-target.csv"%figname))

        if inplace == True:
            self.X_train = X_resample
            self.y_train = y_resample

        return X_resample, y_resample

    def save_dataset(self,foldername, filename):
        """Saves all training, test and original data to csv files in specified folder
        
            Params:
                foldername(str): name of folder created in preprocessed directory
                filename(str): start of filename
                
            Return:
        """
        if not os.path.exists(os.path.join(_preprocesseddir,foldername)):
            os.mkdir(os.path.join(_preprocesseddir,foldername))
        path = os.path.join(_preprocesseddir,foldername)

        self.dataframe.to_csv(os.path.join(path, "%s-original.csv"%filename))
        self.normalised.to_csv(os.path.join(path, "%s-X_train-normalised.csv"%filename))
        self.X_train.to_csv(os.path.join(path, "%s-X_train-tot.csv"%filename))
        self.y_train.to_csv(os.path.join(path, "%s-y_train-tot.csv"%filename))
        self.X_test.to_csv(os.path.join(path, "%s-X_test.csv"%filename))
        self.y_test.to_csv(os.path.join(path, "%s-y_test.csv"%filename))
        
        return

