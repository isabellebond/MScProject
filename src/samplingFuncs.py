import pandas as pd
import os
import numpy as np
import json
from collections import Counter
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from openpyxl import Workbook
import matplotlib.pyplot as plt
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
        self.quantityInit = {}
        self.quantityFin = {}
        self.metrics = {}
        self.impscore = {}
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

        full = self.dataframe.dropna()
        cont = list(set(list(full.select_dtypes(exclude=['object']).columns))-set(['Target']))
        feature_norm = full[cont]/full[cont].max()
        target_norm = full['Target']
        self.features['no_nan'] = feature_norm
        self.target['no_nan'] = target_norm

        
        
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
        self.features['Training_Data'], self.features['Testing_Data'], self.target['Training_Data'], self.target['Testing_Data'] = train_test_split(self.features[data], self.target[data], test_size=testsize, random_state=randomstate)
        
        return
    
    def impute(self, name = 'imputed', data = 'normalised', estimators = BayesianRidge()):

        imp = IterativeImputer(max_iter=100, random_state=0)
        
        
        #self.impscore['original'] = cross_val_score(BayesianRidge(), self.features['no_nan'], self.target['no_nan'], scoring = 'neg_mean_squared_error', cv = 5)
        #impfit = make_pipeline(IterativeImputer(missing_values=np.nan, estimator = estimators), BayesianRidge())
        #self.impscore[name] = cross_val_score(impfit, self.features[data], self.target[data], scoring = 'neg_mean_squared_error', cv = 5)
        impfit = imp.fit_transform(self.features[data])
        self.features[name] = pd.DataFrame(impfit, columns = list(self.features['normalised'].columns))
        self.target[name] = self.target[data]
        self.missingValues[name] = imp.indicator_
        
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
        
        return self.dataframe

    def Bar_Plot(self, data = None, external_data = None):

        if external_data == None:
            df = pd.DataFrame.from_dict(self.metrics).transpose(copy = True)
            print(df)
            if data == 'Test':
                test_cols = [col for col in df.columns if 'test' in col]
                print(test_cols)
                df = df.filter(regex = 'test')    
            if data == 'Train':
                df = df[[df.columns.str.contains('train')]]
            if data == 'All':
                try:
                    df = df.drop(['quantity_initial', 'quantity_final'], axis = 1)
                except KeyError:
                    pass
        
            plt.figure()
            df.plot(kind = 'bar', legend = True, xlabel = 'Sampling Method', ylabel = 'Model Metric')
            plt.savefig(os.path.join(self.dir, 'model_metrics.png'))

        else: 
            df = pd.read_excel(external_data, index_col='Unnamed: 0')
            errors = df.filter(regex = 'error')
            
            #df = df1.filter(items = [col for col in df1.columns if type(col) == float or type(col)==int])
             
            print(df)
            errors = df.loc['Mean Squared Error error']
            #df['Unsampled\nData'] = df1['Unsampled Data']
            df = df.loc['Mean Squared Error']
            print(df)
            #for col in df.index:
               #     colnew = col.replace(' ','\n')
             ##   if type(col) == str:
                #    df[colnew] = df[col]
                 #   errors[colnew] = errors[col]
                  #  df = df.drop(col, axis = 1)
            
            #df = df.filter(regex = 'recall')
            #errors = errors.filter(regex = 'recall')
            print(errors, df)
            
            plt.figure()

            
            df.plot(kind = 'bar', yerr = errors, xlabel = 'Imputation Method', ylabel = 'Mean Squared Error', rot = 90)
            plt.title('Influence of Regression Algorithm on Imputation')
            plt.savefig(os.path.join(self.dir, 'model_metrics.png'), dpi = 600,bbox_inches='tight')
    
    def Scatter_Plot(self, data = None, external_data = None):

        if external_data == None:
            df = pd.DataFrame.from_dict(self.metrics).transpose(copy = True)
            print(df)
            if data == 'Test':
                test_cols = [col for col in df.columns if 'test' in col]
                df = df.filter(items = [col for col in df.columns if type(col) == float or type(col)==int]).transpose(copy = True)
                print(test_cols)
                df = df.filter(regex = 'test')    
            if data == 'Train':
                df = df[[df.columns.str.contains('train')]]
            if data == 'All':
                try:
                    df = df.drop(['quantity_initial', 'quantity_final'], axis = 1)
                except KeyError:
                    pass
        
            plt.figure()
        
            df.plot(legend = True, xlabel = 'Ratio', ylabel = 'Model Metric')
            plt.figure()
            plt.savefig(os.path.join(self.dir, 'model_metrics.png'), dpi = 600)
        
        else:
            df = pd.read_excel(external_data, index_col='Unnamed: 0')
            print(df)
            errors = df.filter(regex = 'error').transpose(copy = True)
            print(errors)
            df = df.filter(items = [col for col in df.columns if type(col) == float or type(col)==int]).transpose(copy = True)
            print(df)

            if data == 'Test':
                df = df.filter(regex = 'test')
                errors = errors.filter(regex = 'test')
            print('here',errors, df)
            
            plt.figure()

            for col in df.columns:
                print(df[col], errors[col])
                plt.errorbar(df.index, df[col], yerr = errors[col])
            plt.ylabel('Model Metric')
            plt.xlabel('Ratio')
            plt.legend(df.columns)
            plt.title('Influence of Ratios: Undersampling')
            plt.savefig(os.path.join(self.dir, 'model_metrics.png'), dpi = 600)
        

        return

        
            
                
    def PCA_plot(self, key = None):
        # Use principal component to condense the 10 features to 2 features
        if not os.path.exists(os.path.join(self.dir,'images')):
            os.makedirs(os.path.join(self.dir,'images'))

        if key == None:
            for key in self.features:
                try:
                    X = self.features[key]
                    y = self.target[key]
                    pca = PCA(n_components=2).fit(X)
                    pca_2d = pca.transform(X)

                    for i in range(0, pca_2d.shape[0]):
                        if y.iloc[i] == 0:
                            c1 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r', marker='+')
                        elif y.iloc[i] == 1:
                            c2 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='b', marker='x')  
    
                    pl.legend([c1, c2], ['Not Digitally Excluded', 'Digitally Exluded'])
                    pl.xlabel('Principle Compenent 1')
                    pl.ylabel('Principle Compenent 2')
                    pl.title(key.replace("_"," ").replace("-",":"))
                    pl.savefig(os.path.join(os.path.join(self.dir,'images'),  "%s.png"%key), dpi = 1000)
                except ValueError:
                    print(key)
        else:
            X = self.features[key]
            y = self.target[key]
            pca = PCA(n_components=2).fit(X)
            pca_2d = pca.transform(X)

            for i in range(0, pca_2d.shape[0]):
                if y.iloc[i] == 0:
                    c1 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r', marker='+')
                elif y.iloc[i] == 1:
                    c2 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='b', marker='x')  
    
            pl.legend([c1, c2], ['Not Digitally Excluded', 'Digitally Exluded'])
            pl.xlabel('Principle Compenent 1')
            pl.ylabel('Principle Compenent 2')
            pl.title(key.replace("_"," ").replace("-",":"))
            pl.savefig(os.path.join(os.path.join(self.dir,'images'), "%s.png"%key), dpi = 1000)

        return

    def tomek_links(self, dataOut = 'Tomek_Links', dataIn = 'Training_Data'):
        """Undersample data according to Tomek Links algorithm
        Plots first two principle components of sampled data
        Returns resampled feature vector and training data
        Resampling according to continuous data only

        Parameters:
                self: used to access dataframe of dataIn
                dataOut: dictionary key resampled data
                dataIn: dictionary key of input data
        
            Return:
                self.features[dataOut]: resampled feature data
                self.target[dataOut]: resampled target data
        """
        sampler = TomekLinks() 

        self.features[dataOut],  self.target[dataOut] = sampler.fit_resample(self.features[dataIn], self.target[dataIn])

        self.quantityInit[dataOut] = Counter(self.target[dataIn])
        self.quantityFin[dataOut] = Counter(self.target[dataOut])
    
        return self.features[dataOut],  self.target[dataOut]

    def near_miss(self, ratio = 0.25, dataOut = 'Near_Miss', dataIn = 'Training_Data'):
        """Undersample data according to Near Miss algorithm
        Plots first two principle components of sampled data
        Returns resampled feature vector and training data
        Resampling according to continuous data only

        Parameters:
                self: used to access dataframe of dataIn
                ratio: output number of samples in minority class / number of samples in majority class
                dataOut: dictionary key resampled data
                dataIn: dictionary key of input data
        
            Return:
                self.features[dataOut]: resampled feature data
                self.target[dataOut]: resampled target data
        """

        sampler = NearMiss(sampling_strategy = ratio) 

        self.features[dataOut],  self.target[dataOut] = sampler.fit_resample(self.features[dataIn], self.target[dataIn])

        self.quantityInit[dataOut] = Counter(self.target[dataIn])
        self.quantityFin[dataOut] = Counter(self.target[dataOut])
        
        return self.features[dataOut],  self.target[dataOut]

    def cluster_centroids(self, ratio = 0.125, dataOut = 'Cluster_Centroid', dataIn = 'Training_Data'):
        """Undersample data according to Cluster centroid algorithm
        Plots first two principle components of sampled data
        Returns resampled feature vector and training data
        Resampling according to continuous data only

        Parameters:
                self: used to access dataframe of dataIn
                ratio: output number of samples in minority class / number of samples in majority class
                dataOut: dictionary key resampled data
                dataIn: dictionary key of input data
        
            Return:
                self.features[dataOut]: resampled feature data
                self.target[dataOut]: resampled target data
        """

        sampler = ClusterCentroids(sampling_strategy = ratio, voting = "hard") 

        self.features[dataOut],  self.target[dataOut] = sampler.fit_resample(self.features[dataIn], self.target[dataIn])
        self.quantityInit[dataOut] = Counter(self.target[dataIn])
        self.quantityFin[dataOut] = Counter(self.target[dataOut])
        

        return self.features[dataOut],  self.target[dataOut]

    def neigbourhood_cleaning(self, dataOut = 'Neighbourhood_Cleaning', dataIn = 'Training_Data'):
        """Undersample data according to Near Miss algorithm
        Plots first two principle components of sampled data
        Returns resampled feature vector and training data
        Resampling according to continuous data only

         Parameters:
                self: used to access dataframe of dataIn
                dataOut: dictionary key resampled data
                dataIn: dictionary key of input data
        
            Return:
                self.features[dataOut]: resampled feature data
                self.target[dataOut]: resampled target data
        """
        sampler = NeighbourhoodCleaningRule()

        self.features[dataOut],  self.target[dataOut] = sampler.fit_resample(self.features[dataIn], self.target[dataIn])
        self.quantityInit[dataOut] = Counter(self.target[dataIn])
        self.quantityFin[dataOut] = Counter(self.target[dataOut])
        

        return self.features[dataOut],  self.target[dataOut]
    
    def SMOTE(self, ratio = 0.25, k = 5, dataOut = "SMOTE", dataIn = 'Training_Data'):
        """Oversample minority data according to SMOTE algorithm
        Plots first two principle components of sampled data
        Returns resampled feature vector and training data
        Resampling includes categorical features

        Parameters:
                self: used to access dataframe of dataIn
                ratio: output number of samples in minority class / number of samples in majority class
                k: number of referenc
                dataOut: dictionary key resampled data
                dataIn: dictionary key of input data

        
        Return:
                self.features[dataOut]: resampled feature data
                self.target[dataOut]: resampled target data
        """

        sampler = SMOTE( sampling_strategy = ratio, k_neighbors = k) 

        self.features[dataOut],  self.target[dataOut] = sampler.fit_resample(self.features[dataIn], self.target[dataIn])
        self.quantityInit[dataOut] = Counter(self.target[dataIn])
        self.quantityFin[dataOut] = Counter(self.target[dataOut])

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
            y_pred_test = lr.predict(self.features['Testing_Data'])
            y_test = self.target['Testing_Data']
            y_pred_train = lr.predict(self.features['Training_Data'])
            y_train = self.target['Training_Data']

            metrics = {}
            metrics['roc_auc_test'] = roc_auc_score(y_test, y_pred_test)
            metrics['roc_auc_train'] = roc_auc_score(y_train, y_pred_train)
            metrics['f1_test'] = f1_score(y_test, y_pred_test)
            metrics['f1_train'] = f1_score(y_train, y_pred_train)
            metrics['recall_test'] = recall_score(y_test, y_pred_test)
            metrics['recall_train'] = recall_score(y_train, y_pred_train)
            try:
                metrics['quantity_initial'] = self.quantityInit[key]
                metrics['quantity_final'] = self.quantityFin[key]
            except KeyError:
                pass

            self.metrics[key] = metrics
        
        return
    
    def save_results(self, prefix, data = 'all'):

        wb = Workbook()
        wb.save(filename = os.path.join(self.dir, '%s_creation_metrics12.xlsx'%prefix))

        if data == 'all':
            for key in self.results:
                df = pd.DataFrame.from_dict(self.results[key])
                with pd.ExcelWriter(os.path.join(self.dir, '%s_creation_metrics12.xlsx'%prefix), engine="openpyxl", mode = 'a') as writer:
                    df.to_excel(writer, sheet_name = key)
        else:
            df = pd.DataFrame.from_dict(self.results[data])
            with pd.ExcelWriter(os.path.join(self.dir, '%s_creation_metrics12.xlsx'%prefix), engine="openpyxl", mode = 'a') as writer:
                df.to_excel(writer, sheet_name = data)
        
        
        dataframe = pd.DataFrame.from_dict(self.metrics)    
        dataframe.to_excel(os.path.join(self.dir, '%s_model_metrics12.xlsx'%prefix))
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

