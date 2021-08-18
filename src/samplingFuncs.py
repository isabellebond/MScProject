import pandas as pd
import os
import numpy as np
import json
from collections import Counter
import joblib
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from openpyxl import Workbook
import pylab as pl
from sklearn.metrics import f1_score, recall_score, roc_auc_score

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
    def __init__(self, dataframe, directory):
        self.dataframe = dataframe
        self.dir = directory
        self.target = {}
        self.features = {}
        self.results = {}
        self.models = {}
        self.missingValues = {}
        self.quantityChange = {}
        self.metrics = {}
        self.target['original'] = self.dataframe['Target']
        self.features['original'] = self.dataframe.drop('Target', axis = 1)
    
    def normalise(self):
        """
        Create new dataframe with all continuous features normalised.

            Parameters:
                dataframe
        
            Return:
                normdata: dataframe with normalised column values
        """
        cont = list(set(list(self.dataframe.select_dtypes(exclude=['object']).columns))-set(['Target']))
        feature_norm = self.dataframe[cont]/self.dataframe[cont].max()
        target_norm = self.target['original']/self.target['original'].max()
        
        self.features['normalised'] = feature_norm
        self.target['normalised'] = target_norm

        return
    
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

    def train_test_split(self, testsize, randomstate = None, data = 'imputed'):
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
        self.features['train_test_split'], self.features['test_data'], self.target['train_test_split'], self.target['test_data'] = train_test_split(self.features[data], self.target[data], test_size=testsize, random_state=randomstate)
        
        return
    
    def impute(self, name = 'imputed', data = 'normalised'):

        imp = IterativeImputer(max_iter=100, random_state=0)
        impfit = imp.fit_transform(self.features[data])
    
        self.features[name] = pd.DataFrame(impfit, columns = list(self.features['normalised'].columns))
        self.target[name] = self.target[data]
        self.missingValues[name] = imp.indicator_
        
        return 

    def data_split(self, ratio = 0.25, dataIn = 'train_test_split', dataOut = 'split data'):
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
        for i in range(0,len(self.target[dataIn])):
            if self.target[dataIn][i] == 0:
                class1.append(i)
            else:
                class2.append(i)

        X_class1 = self.features[dataIn].iloc[class1,:]
        X_class2 = self.features[dataIn].iloc[class2,:]
        y_class2 = np.ones((len(class2),1))

        if ratio == 0.25:
            Xa,Xb = train_test_split(X_class1, test_size = 0.5)
            X1, X2 = train_test_split(Xa, test_size = 0.5)
            X3, X4 = train_test_split(Xb, test_size = 0.5)

            X = [X1,X2,X3,X4]
            k = 1
            for item in X:
                y_class1 = np.zeros((len(item),1))
                self.target['%s_%s'%(dataOut, k)] = pd.DataFrame(data = np.append(y_class1, y_class2, axis = 0), columns = ['Target'])
                X_new = np.append(item, X_class2, axis = 0)
                self.features['%s_%s'%(dataOut, k)] = pd.DataFrame(data = X_new, columns = self.features[dataIn].columns)
                k+=1
                #df[column] = df[column].apply( lambda x: np.nan if x == self.X_mean.get(column) else x)
        
        elif ratio == 0.5:
            Xa,Xb = train_test_split(X_class1, test_size = 0.5)

            X = [Xa,Xb]
            k = 1
            for item in X:
                y_class1 = np.zeros((len(item),1))
                self.target['%s_%s'%(dataOut, k)] = pd.DataFrame(data = np.append(y_class1, y_class2, axis = 0), columns = ['Target'])
                X_new = np.append(item, X_class2, axis = 0)
                self.features['%s_%s'%(dataOut, k)] = pd.DataFrame(data = X_new, columns = self.features[dataIn].columns)
                k+=1
        
        elif ratio == 0.125: 
            Xa,Xb = train_test_split(X_class1, test_size = 0.5)

            Xz, Xy = train_test_split(Xa, test_size = 0.5)
            Xx, Xw = train_test_split(Xb, test_size = 0.5)

            X1, X2 = train_test_split(Xz, test_size = 0.5)
            X3, X4 = train_test_split(Xy, test_size = 0.5)
            X5, X6 = train_test_split(Xx, test_size = 0.5)
            X7, X8 = train_test_split(Xw, test_size = 0.5)

            X = [X1,X2,X3,X4,X5,X6,X7,X8]
            k = 1
            for item in X:
                y_class1 = np.zeros((len(item),1))
                self.target['%s_%s'%(dataOut, k)] = pd.DataFrame(data = np.append(y_class1, y_class2, axis = 0), columns = ['Target'])
                X_new = np.append(item, X_class2, axis = 0)
                self.features['%s_%s'%(dataOut, k)] = pd.DataFrame(data = X_new, columns = self.features[dataIn].columns)
                k+=1
        
        else:
            raise ValueError('Unaccepted value for ratio. Ratio must be 0.125, 0.25 or 0.5.')

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

    def tomek_links(self, dataOut = 'Tomeklinks', dataIn = 'train_test_split'):
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
        sampler = TomekLinks() 

        self.features[dataOut],  self.target[dataOut] = sampler.fit_resample(self.features[dataIn], self.target[dataIn])

        self.quantityChange[dataOut] = Counter(self.target[dataIn]) - Counter(self.target[dataOut])
    
        return 

    def near_miss(self, dataOut = 'NearMiss', dataIn = 'train_test_split'):
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

        sampler = NearMiss() 

        self.features[dataOut],  self.target[dataOut] = sampler.fit_resample(self.features[dataIn], self.target[dataIn])

        self.quantityChange[dataOut] = Counter(self.target[dataIn]) - Counter(self.target[dataOut])
        
        return 

    def cluster_centroids(self, ratio = 0.125, dataOut = 'ClusterCentroid', dataIn = 'train_test_split'):
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

        sampler = ClusterCentroids(sampling_strategy = ratio, voting = "hard") 

        self.features[dataOut],  self.target[dataOut] = sampler.fit_resample(self.features[dataIn], self.target[dataIn])
        self.quantityChange[dataOut] = Counter(self.target[dataIn]) - Counter(self.target[dataOut])
        

        return

    def neigbourhood_cleaning(self, dataOut = 'Neighbourhood Cleaning', dataIn = 'train_test_split'):
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
        sampler = NeighbourhoodCleaningRule()

        self.features[dataOut],  self.target[dataOut] = sampler.fit_resample(self.features[dataIn], self.target[dataIn])
        self.quantityChange[dataOut] = Counter(self.target[dataIn]) - Counter(self.target[dataOut])
        

        return
    
    def SMOTE(self, ratio = 0.25, k = 5, dataOut = "SMOTE", dataIn = 'train_test_split'):
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

        sampler = SMOTE( sampling_strategy = ratio, k_neighbors = k) 

        self.features[dataOut],  self.target[dataOut] = sampler.fit_resample(self.features[dataIn], self.target[dataIn])
        self.quantityChange[dataOut] = Counter(self.target[dataIn]) - Counter(self.target[dataOut])

        return 

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

    def logistic_regression(self, C = [0.001,0.01,0.1,1,10,1000], dataIn = 'all'): 
        lr = LogisticRegression(max_iter=1000)
        params = {
            'C' : C
        }
        lr_CV = GridSearchCV(lr, params, cv=5)
        if dataIn == 'all':
            for key in self.features:
                try:
                    lr_CV.fit(self.features[key], self.target[key])
        
                    self.results[key] = lr_CV.cv_results_
                    self.models[key] = lr_CV.best_estimator_
                    print('yes', self.models[key])
                except ValueError:
                    pass
        
        else:
            lr_CV.fit(self.features[dataIn], self.target[dataIn])
        
            self.results[dataIn] = lr_CV.cv_results_
            self.models[dataIn] = lr_CV.best_estimator_

        return
    
    def test_model(self):

        for key in self.models:
            lr = self.models[key].fit(self.features[key], self.target[key])
            y_pred_test = lr.predict(self.features['test_data'])
            y_test = self.target['test_data']
            y_pred_train = lr.predict(self.features['train_test_split'])
            y_train = self.target['train_test_split']

            metrics = {}
            metrics['roc_auc_test'] = roc_auc_score(y_test, y_pred_test)
            metrics['roc_auc_train'] = roc_auc_score(y_train, y_pred_train)
            metrics['f1_test'] = f1_score(y_test, y_pred_test)
            metrics['f1_train'] = f1_score(y_train, y_pred_train)
            metrics['recall_test'] = recall_score(y_test, y_pred_test)
            metrics['recall_train'] = recall_score(y_train, y_pred_train)
            try:
                metrics['quantity_change'] = self.quantityChange[key]
            except KeyError:
                pass

            self.metrics[key] = metrics
            print(self.metrics)
        
        return
    
    def save_results(self, prefix, data = 'all'):

        wb = Workbook()
        wb.save(filename = os.path.join(self.dir, '%s_creation_metrics.xlsx'%prefix))

        if data == 'all':
            for key in self.results:
                df = pd.DataFrame.from_dict(self.results[key])
                with pd.ExcelWriter(os.path.join(self.dir, '%s_creation_metrics.xlsx'%prefix), engine="openpyxl", mode = 'a') as writer:
                    df.to_excel(writer, sheet_name = key)
        else:
            df = pd.DataFrame.from_dict(self.results[data])
            with pd.ExcelWriter(os.path.join(self.dir, '%s_creation_metrics.xlsx'%prefix), engine="openpyxl", mode = 'a') as writer:
                df.to_excel(writer, sheet_name = data)
        
        
        dataframe = pd.DataFrame.from_dict(self.metrics)    
        dataframe.to_excel(os.path.join(self.dir, '%s_model_metrics.xlsx'%prefix))
        return

    def save_dataset(self, prefix, data = 'all'):

        """Saves all training, test and original data to csv files in specified folder
        
            Params:
                foldername(str): name of folder created in preprocessed directory
                filename(str): start of filename
                
            Return:
        """

        wb = Workbook()
        wb.save(filename = os.path.join(self.dir,'%s_features.xlsx'%prefix))
        wb.save(filename = os.path.join(self.dir,'%s_target.xlsx'%prefix))

        if data == 'all':
            for key in self.features:
                df_features = self.features[key]
                df_target = self.target[key]
                with pd.ExcelWriter(os.path.join(self.dir,'%s_features.xlsx'%prefix), engine="openpyxl", mode = 'a') as writer:
                    df_features.to_excel(writer, sheet_name = key)
                with pd.ExcelWriter(os.path.join(self.dir,'%s_target.xlsx'%prefix), engine="openpyxl", mode = 'a') as writer:
                    df_target.to_excel(writer, sheet_name = key)
        else:
            df_features = self.features[data]
            df_target = self.target[data]
            with pd.ExcelWriter(os.path.join(self.dir,'%s_features.xlsx'%prefix), engine="openpyxl", mode = 'a') as writer:
                df_features.to_excel(writer, sheet_name = data)
            with pd.ExcelWriter(os.path.join(self.dir,'%s_target.xlsx'%prefix), engine="openpyxl", mode = 'a') as writer:
                df_target.to_excel(writer, sheet_name = data)

        return

    def save_models(self, folder = 'models', model = 'all'):
        path = os.path.join(self.dir, folder)

        if not os.path.exists(path):
            os.makedirs(path)
        
        if model == 'all':
            for key in self.models:
                joblib.dump(self.models[key], os.path.join(path, "%s_model.pkl"%key ))
        return
    
    def ada_boost(self):
        pass
