# -*- coding: utf-8 -*-
"""
DataProcessor.
@author: neeraj kesavan
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

class DataProcessor:
    """Class DataPreprocessor. To process raw data and make it ready for model training and prediction.
    Feature selection, feature extraction, outlier detection and removal, and unskewing are performed on data.
    DataProcessor has the following properties

    Attributes:
        train_data_org: raw training data
        test_data_org: raw testing data
        target_var: target variable name
        train_output_loc: file name + path to which final processed train data to be saved.
        test_output_loc: file name + path to which final processed test data to be saved.
        predictor_var: list of predictor variable names.
        numVariable: continuous varable name in dataset.
        catVariable: catagorical variable names in the dataset.
    
    Methods:
        __init__(): Constructor, Populates required data.
        preprocess(): Performs cleansing, feature seletion and extraction.
        populate_predictors_bytype: populates predictor variables based on their data type.
        plot_data(): Plots data for manual outlier detection.
        update_predictor_list: Updates the predictor list based on the changes in training data
        finalize_data(): finalizes the processed data and saves the final data.
        detect_outliers(): detects outliers in dataset and returns them based on the threshold provided.
        get_features_on_variance(): Identifies features with low or no variance.        
        unskew_data(): performs log transformations to unskew data based on the threshold provided.
        run(): performs the data processing by using the above methods as required.
    
    """
    train_data_org = pd.DataFrame()
    test_data_org = pd.DataFrame()
    target_var = ''
    train_output_loc = ''
    test_output_loc= ''
    predictor_var = []
    numVariable = []
    catVariable = [] 
    
    
    def __init__(self, train_file_loc='../resources/train.csv', test_file_loc='../resources/test.csv', target_var='SalePrice', train_output_loc='../resources/train_final.csv', test_output_loc='../resources/test_final.csv'):
        """Takes arguments 'train_file_loc', 'test_file_loc', 'train_output_loc', 'test_output_loc', 'target_var'
        train_file_loc: path to raw training data set. Default='resources/data/train.csv'
        test_file_loc: path to raw testing data set. Default='resources/data/test.csv'
        train_output_loc=output file name + path. Default='resources/data/train_final.csv'
        test_output_loc=output file name + path. Default='resources/data/test_final.csv'
        target_var: target variable name. Default='SalePrice'
        
        Constructor. Initializes the class variable by populating the required data
        """
        #Loads raw train and test data        
        self.train_data_org = pd.read_csv(train_file_loc)
        self.test_data_org = pd.read_csv(test_file_loc)
        
        #Populates predictor varible names.
        self.predictor_var = self.train_data_org.columns.tolist()
        
        #Setting target variable name.
        self.target_var = target_var
        
        #populates predicator varible name by data type.
        self.populate_predictors_bytype()
        
        #Setting output file location
        self.train_output_loc = train_output_loc
        self.test_output_loc = test_output_loc
    
    def preprocess(self):
        """ Takes arguents 'train_path' and 'test_path'
        train_path: path to training data set
        test_path: path to testing data set
        
        Performs cleansing, feature selection and extraction.    
        Saves the preprocessed data.
        """
        
        #Dropping Id (Id on tet data is required to create submisson file)
        self.train_data_org.drop(['Id'], axis=1, inplace=True)
        #self.test_data_org.drop(['Id'], axis=1, inplace=True)
        
        train = self.train_data_org.copy()
        test = self.test_data_org.copy()
        
        #Dropping features which have more than 50% missing values
        train.drop(['Fence', 'Alley','MiscFeature', 'PoolQC' ], axis=1, inplace=True)
        test.drop(['Fence', 'Alley','MiscFeature', 'PoolQC' ], axis=1, inplace=True)
        
        #New feature created 'FirePlacePresent'. Many houses may not have a fire place.
        train['FirePlacePresent'] = train['FireplaceQu'].apply(lambda x: 0 if pd.isna(x) else 1)
        test['FirePlacePresent'] = test['FireplaceQu'].apply(lambda x: 0 if pd.isna(x) else 1)
        
        #Imputing values for missing data
        train['FireplaceQu'] = train['FireplaceQu'].fillna('None')  
        test['FireplaceQu'] = test['FireplaceQu'].fillna('None')
        
        #Imputing values for LotFrontage based on the Neighborhood. 
        #Range LotFrontage values depends on the Neighborhood.
        train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
        test['LotFrontage'] = test.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
        
        #All values are NaN for those houses which doesn't have garage.
        #GarageArea  and GarageCars are zero.
        train['GarageYrBlt'] = train['GarageYrBlt'].fillna(0)
        train['GarageCond'] = train['GarageCond'].fillna('None')
        train['GarageFinish'] = train['GarageFinish'].fillna('None')
        train['GarageQual'] = train['GarageQual'].fillna('None')
        train['GarageType'] = train['GarageType'].fillna('None')
        #new feature 'GaragePresent' is created. There are houses without garage
        train['GaragePresent'] = train['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
        test['GarageYrBlt'] = test['GarageYrBlt'].fillna(0)
        test['GarageCond'] = test['GarageCond'].fillna('None')
        test['GarageFinish'] = test['GarageFinish'].fillna('None')
        test['GarageQual'] = test['GarageQual'].fillna('None')
        test['GarageType'] = test['GarageType'].fillna('None')
        #new feature 'GaragePresent' is created. There are houses without garage
        test['GaragePresent'] = test['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
        
        #Here we can create one extra feature GaragePresent
#        train['GaragePresent'] = train['GarageArea'].apply(lambda x: 0 if x is 'None' else 1)
#        test['GaragePresent'] = test['GarageArea'].apply(lambda x: 0 if x is 'None' else 1)
        ##No improvement on model
        
        #impute 'None' for missing values
        train['BsmtFinType1'] = train['BsmtFinType1'].fillna('None')
        train['BsmtFinType2'] = train['BsmtFinType2'].fillna('None')
        train['BsmtQual'] = train['BsmtQual'].fillna('None')
        train['BsmtExposure'] = train['BsmtExposure'].fillna('None')
        train['BsmtCond'] = train['BsmtCond'].fillna('None')
        #new feature 'BsmtPresent' is created. There are houses without basement
        train['BsmtPresent'] = train['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
        test['BsmtFinType1'] = test['BsmtFinType1'].fillna('None')
        test['BsmtFinType2'] = test['BsmtFinType2'].fillna('None')
        test['BsmtQual'] = test['BsmtQual'].fillna('None')
        test['BsmtExposure'] = test['BsmtExposure'].fillna('None')
        test['BsmtCond'] = test['BsmtCond'].fillna('None')
        #new feature 'BsmtPresent' is created. There are houses without basement.
        test['BsmtPresent'] = test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
        
        #MasVnrArea, MasVnrType
        #impute zero and 'None'
        train['MasVnrArea'] = train['MasVnrArea'].fillna(0)
        train['MasVnrType'] = train['MasVnrType'].fillna('None')
        test['MasVnrArea'] = test['MasVnrArea'].fillna(0)
        test['MasVnrType'] = test['MasVnrType'].fillna('None')
        
        #Imputing missing 'MSZoning' values based on 'MSSubClass'.
        #Range of 'MSZoning' values depends on 'MSSubClass'
        test['MSZoning'] = test.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

        #Creating new features 'BuildingAge and 'BuildingGarageAge'
        train['BuildingAge'] = train['YearBuilt'].apply(lambda x:2019-x)        
        test['BuildingAge'] = test['YearBuilt'].apply(lambda x:2019-x)
        
        train['BuildingGarageAge'] = train['GarageYrBlt'].apply(lambda x: (2019-x) if x>0 else x)
        test['BuildingGarageAge'] = test['GarageYrBlt'].apply(lambda x: (2019-x) if x>0 else x)


#        ##No improvement on model bu using below new features.
   
#        train['BuildingRemodAddAge'] = train['YearRemodAdd'].apply(lambda x:2019-x)
#        test['BuildingRemodAddAge'] = test['YearRemodAdd'].apply(lambda x:2019-x)        
#
#        train['BuildingSoldAge'] = train['YrSold'].apply(lambda x:2019-x)
#        test['BuildingSoldAge'] = test['YrSold'].apply(lambda x:2019-x)

#        train.drop(['YrSold', 'YearRemodAdd', 'GarageYrBlt', 'YearBuilt'], axis=1, inplace=True)
#        test.drop(['YrSold', 'YearRemodAdd', 'GarageYrBlt', 'YearBuilt'], axis=1, inplace=True)        
        
        #Electrical
        #train['Electrical'].mode()  is 'SBrkr'
        train['Electrical'] = train['Electrical'].fillna('SBrkr')
        
        #Updating predictor list
        self.update_predictor_list(train) 
        
        #impute values for remaining missing data
        for i in self.catVariable:
            train[i] = train[i].fillna('None')
            test[i] = test[i].fillna('None')
        for i in self.numVariable:
            train[i] = train[i].fillna(0)
            test[i] = test[i].fillna(0)
        
        
        return train, test
        
    def populate_predictors_bytype(self):
        """Takes no arguments 
               
        populate predictor variables 'catVariable' and 'numVariable' based on the datatype.
        """
        self.catVariable = []
        self.numVariable = []
        
        for i in self.train_data_org.columns:
            if self.train_data_org[i].dtype in ['object']:
                self.catVariable.append(i)
            else:
                self.numVariable.append(i)
        #Removing target variable and Id        
        if(self.target_var in self.numVariable):
            self.numVariable.remove(self.target_var)
            
        if('Id' in self.numVariable):
            self.numVariable.remove('Id')
            

    def plot_data(self, train, X, Y='SalePrice', no_var=1):
        """Takes arguments 'train', 'X', 'Y' and 'no_var'.
        train: training data
        X: x axis column
        Y: y axis column. Default='SalePrice'
        no_var: no. of variables for plotting. Default=1 (only 1 or 2)
        
        Plots the data based on input. Can be used for manual outlier detection.
        """
        
        if(no_var==1):
            sns.distplot(a = train[Y])
            plt.show()
            plt.gcf().clear()
        else:
            sns.lmplot(data=train, x=X , y=Y)
            plt.show()
            plt.gcf().clear()
    
    def update_predictor_list(self, train):
        """Takes argument 'train'
        
        Updates the class variable based on the changes in training data
        """
        #Updating predictor list
        self.train_data_org = train.copy()
        self.populate_predictors_bytype()
        self.predictor_var = train.columns.tolist() 
    
    def finalize_data(self, train, test):
        """Takes arguments 'train_path' and 'test_path'
        train_path: path to processed training data set
        test_path: path to processed testing data set
        
        Creates dummy variables and saves the final data set as 'train_final.csv' and 'test_final.csv'.
        """
        
        #Combines dataset for creating dummy varibles for categorigal features.
        train['dataTag']= 'Train'
        test['dataTag'] = 'Test'
        allData=pd.concat([train, test],ignore_index=True)
        
        allData[self.target_var] = allData[self.target_var].fillna(0)   
        allData['Id'] = allData['Id'].fillna(0)
        
        #Creating dummy variables.
        finalAllData =  pd.get_dummies(allData).reset_index(drop=True)

        #Splitting train and test data.
        finalTrain = finalAllData[finalAllData.dataTag_Train==1]
        finalTest = finalAllData[finalAllData.dataTag_Test==1]
        
        #Removing irrelevant coulmns.
        finalTrain.drop(['dataTag_Train'], axis=1, inplace=True)
        finalTrain.drop(['dataTag_Test'], axis=1, inplace=True)
        finalTrain.drop(['Id'], axis=1, inplace=True)
        finalTest.drop(['dataTag_Train'], axis=1, inplace=True)
        finalTest.drop(['dataTag_Test'], axis=1, inplace=True)
        finalTest.drop([self.target_var], axis=1, inplace=True)
        
        #Saving final train and test data.
        finalTrain.to_csv(self.train_output_loc,index=False)
        finalTest.to_csv(self.test_output_loc,index=False)
        
        print("Data processing completed...")

    def detect_outliers(self, train, threshold=0.001):
        """Takes arguments 'train' and 'threshold'.
        train: training data.
        threshold: threshold value by which outlier to be returned. Default=0.001
        
        Identifies outliers using 'OLS - bonf(p)'.
        Returns 'bonf_outlier' based on the threshold.
        """
        train_X = train.copy()
        train_X.drop(['SalePrice'], axis = 1, inplace=True)   
               
        for i in self.numVariable:
            train_X[i] = train_X[i].fillna(0)

        model =sm.OLS(train['SalePrice'], train_X[self.numVariable] ) 
        results = model.fit()
        #Outlier test
        bonf_test = results.outlier_test()['bonf(p)']
        #filter outliers based on threshold
        bonf_outlier = list(bonf_test[bonf_test<threshold].index) 
        
        return bonf_outlier
    
    def get_features_on_variance(self, train_data, variance=0.3):
        """Takes arguments 'train_data' and 'variance'.
        train_data: training data
        variance: variance threshold by which features should be returned. Default=0.3

        Filters low variance or no variance features based on the the threshold variance given.
        
        Returns 'feature_list'.
        """
        train = train_data.copy()
        train.drop('SalePrice', axis=1, inplace=True)
        feature_variance = train.var().sort_values(ascending=False)
        #Filters feature based on the variance threshold.
        feature_list = feature_variance[feature_variance<variance].index.tolist()
        
        return feature_list
        
    def unskew_data(self, train, test, columns, threshold=1):
        """Takes arguments 'train', 'test', 'columns' and 'threshold'
        train: training data
        test: testng data
        columns: continuous predictor variables.
        threshold: threhold by which columns selected based on skewness. Default=1
        
        Apply log transformations to the selected skewed column values.
        Returns unskewed 'train' and 'test' data
        """
        skew_data = train[columns].skew() 
        #Selects skewed coulmns based on threshold  for transformation
        skewed_columns = skew_data[skew_data>threshold].index.tolist()               
        
        #Performs log transformations.
        for i in skewed_columns:
            train[i] = np.log1p(train[i])
            test[i] = np.log1p(test[i])
        
        return train, test
       
    def run(self):
        """Takes arguents 'train_path' and 'test_path'
        train_path: path to training data set
        test_path: path to testing data set
        
        Performs the relevant required data processing steps in the specified order.
        """
        print("Data processing started...")
        #Preprocesing
        train, test = self.preprocess()   

        #Updating predictor list
        self.update_predictor_list(train)        
        
        #Performs outlier detection.
        outliers = self.detect_outliers(train)
        
        #removes outliers.
        train.drop(outliers, axis=0, inplace=True)
        
        #Selects features with low or no varince
        low_var_features = self.get_features_on_variance(train, variance=0.3)
        
        #Removes low variance features.
        train.drop(low_var_features, axis=1, inplace=True)
        test.drop(low_var_features, axis=1, inplace=True)

        #Updating predictor list
        self.update_predictor_list(train)
        
        #Unskewing data
        train, test = self.unskew_data(train, test, self.numVariable, threshold=1)        
      
        #Finalizing data.
        self.finalize_data(train, test)
