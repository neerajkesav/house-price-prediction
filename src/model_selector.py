# -*- coding: utf-8 -*-
"""
ModelSelector.
@author: neeraj kesavan
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import matplotlib.pylab as plt

from matplotlib.pylab import rcParams
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from ensemble_regressor import EnsembleRegressor


class ModelSelector:
    """
    Class ModelSelector. Performs model selection process.

    Attributes:
        output_file_path: filename + path to save the final model.
    
    Methods:
        cross_validate(): Performs cross validation of the model and print mean validation score.
        finalize_and_save(): fits and saves the final model.
        get_feat_imp(): To get the feature importance of a model using optimal parameter.
        get_predictors_feat_imp(): To get predictors from feature importance based on threshold.
        get_predictors(): To get predictor variables to be used with model.
        run(): Performs models selection process on the specified order.
    
    """
    output_file_path=''
    target_var = ''
    def __init__(self, target_var='SalePrice', output_file_path='../model/model_final'):
        """Takes argument 'output_file_path', 'target_var'.
        output_file_path: ilename + path to save the final model. Default='resources/model/model_final'
        target_var: target variable name. Default='SalePrice'
        
        Constructor Initialize class attributes.
        """
        self.output_file_path = output_file_path
        self.target_var = target_var
        
    def cross_validate(self, model, X, y, cv=5, n_jobs=-1):
        """ Takes arguments 'model', 'X', 'y', 'cv', 'n_jobs'.
        model: model to be corss validated.
        X: predictor data.
        y: target data.
        cv: cv fold value. Default=5
        n_jobs: no. of threads. Default=-1 (automatically detects threads present).
        
        Performs cross validation of the model and print mean validation score.
        """
        cv_score = cross_val_score(estimator=model, X=X, y=y, cv = cv, n_jobs=n_jobs)
        print('CV Score is: '+ str(np.mean(cv_score)))

    def finalize_and_save(self, model, filename, input_train, output_train):
        """Takes arguments 'model', 'filename', 'input_train', 'output_train'.
        model: finalized model.
        filename: filename to which model to be saved.
        input_train: input part of train_set.
        output_train: output part of train_set.
        
        Saves the model to disk.
        """
        model.fit(input_train, output_train)
        #Save the model to disk
        pickle.dump(model, open(filename, 'wb' ))
        print("\nModel is saved..\n")
    
    def get_feat_imp(self, model, train, predictor_vars, cv_folds=5, early_stopping_rounds=50, plot=False):
        """Takes arguments 'model', 'train', 'predictor_vars', 'target_var', 'cv_folds', 'early_stopping_rounds','plot'
        model: xgboost model to be used to get feature importance
        train: training data
        predictor_vars: predictor variables.
        target_var: target varable
        cv_folds: no. of cv folds. Default=5
        early_stopping_rounds: Early stopping in case of no improvement. Default=50.
        plot: if 'True' plots the feature importance.Default=False.
        
        To get the feature importance of a model using optimal parameter.
        Plots the feature importance if specified.
        
        Returns 'feat_imp'
        """
        #get current params
        xgb_param = model.get_xgb_params()
        #Evaluates the model for optimal performance
        xgtrain = xgb.DMatrix(train[predictor_vars].values, label=train[self.target_var].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=model.get_params()['n_estimators'], nfold=cv_folds,
            metrics='rmse', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        #Sets the optimal params
        model.set_params(n_estimators=cvresult.shape[0])
    
        #Fit the model
        model.fit(train[predictor_vars], train[self.target_var],eval_metric='rmse')
        #gets feature importance
        feat_imp = pd.Series(model.get_booster().get_fscore()).sort_values(ascending=False)
        
        #Plots feature importance if 'True'
        if(plot):
            rcParams['figure.figsize'] = 12, 4
            feat_imp.plot(kind='bar', title='Feature Importances')
            plt.ylabel('Feature Importance Score')
            plt.gcf().clear()
            
        #returns feature importance
        return feat_imp
        
    def get_predictors_feat_imp(self, feature_imp, threshold=5):
        """Takes argument 'feature_imp' and 'threshold'.
        feature_imp: feature importance
        threshold: threshold value by which feature_imp to be filtered.
        
        To get predictors from feature importance based on threshold.
        
        Returns 'predictors' variable names as a list
        """
        predictors = feature_imp[feature_imp>threshold].index.tolist()
        
        return predictors
        
    def get_predictors(self, train):
        """Takes argument 'train'.
        train: training data.
        
        To get predictor variables to be used with model.        
        Returns the predictor variable names as a list.
        """
        predictors =  train.columns.tolist()
        
        if(self.target_var in predictors):
            predictors.remove(self.target_var)
            
        if('Id' in predictors):
            predictors.remove('Id')
        
        return predictors

    
    def run(self, train_data_path):  
        """Takes argument 'train_data_path'.
        train_data_path: Training data path.
        
        Performs models selection process on the specified order.
        A no. of reqred models can added to this method body and corss validated
        These can be saved as it is or ensembling can be applied. 
        """
        #Loading training data
        dtrain = pd.read_csv(train_data_path)
        #gets predictors
        predictor_vars = self.get_predictors(dtrain)

        #Model I
        xgboost = XGBRegressor(learning_rate=0.06,n_estimators=1000,
                                     max_depth=2, min_child_weight=2,
                                     gamma=0, subsample=0.4,
                                     colsample_bytree=0.2,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=77)
        
        #Model II
        xgboost2 = XGBRegressor(learning_rate=0.04,n_estimators=1500,
                                     max_depth=2, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.2,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=99,
                                     reg_alpha=1.7)
        
        #Model III
        xgboost3 = XGBRegressor(learning_rate=0.02,n_estimators=1200,
                                     max_depth=3, min_child_weight=2,
                                     gamma=0, subsample=0.65,
                                     colsample_bytree=0.2,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=585,
                                     reg_alpha=5000)
            
        #Model IV                         
        lightgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       min_data_in_leaf=5,
                                       learning_rate=0.02, 
                                       n_estimators=3000,
                                       max_bin=320, 
                                       bagging_fraction=0.85,
                                       bagging_freq=10, 
                                       bagging_seed=9,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=9,
                                       data_random_seed=9,
                                       reg_alpha=0.55,
                                       reg_lambda=0.3,
                                       verbose=-1)  
           
        #Model V                    
        lightgbm2 = LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       min_data_in_leaf=3,
                                       learning_rate=0.01, 
                                       n_estimators=4000,
                                       max_bin=295, 
                                       bagging_fraction=0.5,
                                       bagging_freq=10, 
                                       bagging_seed=24,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=24,
                                       data_random_seed=24,
                                       reg_alpha=10,
                                       reg_lambda=0.7,
                                       verbose=-1)
   
        #Ensembling all the five models
        ens_model = EnsembleRegressor([xgboost, xgboost2, xgboost3, lightgbm, lightgbm2])
        
        #Performs cross validation on the ensembled model.
        self.cross_validate(cv=5,model=ens_model,X=dtrain[predictor_vars], y=dtrain[self.target_var], n_jobs = 1)
        #CV Score is: 0.92528287952747 all predictors
        
        #Saving the final model.
        self.finalize_and_save(ens_model, self.output_file_path, dtrain[predictor_vars], dtrain[self.target_var] )
        