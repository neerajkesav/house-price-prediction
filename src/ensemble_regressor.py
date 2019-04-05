# -*- coding: utf-8 -*-
"""
EnsembleRegressor.
@author: neeraj kesavan
"""
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin

class EnsembleRegressor(BaseEstimator, RegressorMixin):
    """Class EnsembleRegressor. To perform ensembling of models.
    Inherits BaseEstimator, RegressorMixin.
    
    Methods:
        __init__(): Constructor. Initialize class instance with regressors.
        fit(): fits training data to the model.
        predict(): performs prediction using all the models and returns the mean prediction.
    
    """
    def __init__(self, regressor_models=None):
        """Takes argument 'regressors'
        regressor_models: list of models.
        
        Initialize class instance with regressors.
        """
        self.regressor_models = regressor_models

    def fit(self, train_X, train_y):
        """Takes arguments 'train_X' and 'train_y'
        train_X: predictors of train  data
        train_y: target data
        
        Fits training data to the model.
        """
        for model in self.regressor_models:
            model.fit(train_X, train_y)

    def predict(self, test_X):
        """Takes argument test_X.
        test_X: test data on which prediction is to be performed.
        
        Performs prediction using each model and returns the mean prediction.
        """
        self.predictions_list = list()
        for model in self.regressor_models:
            self.predictions_list.append((model.predict(test_X).ravel()))
        return (np.mean(self.predictions_list, axis=0))

