# -*- coding: utf-8 -*-
"""
ParameterTuner.
@author: neeraj kesavan
"""
from sklearn.model_selection import GridSearchCV

class ParameterTuner:
    """Class ParameterTuner. performs hyperparameter tuning using GridSearchCV.
    
    Methods:
        tune_parameters(): Performs parameter tuning, print and returns best parameters.
    """
    
    def tune_parameters(self, model, param_set, train, predictor_var, target_var):
        """Takes arguments 'model', 'param_set', 'train', 'predictor_var' and 'target_var'.
        model: model to be used for parameter tuning
        param_set: dict of parameter to be tuned.
        train: training data.
        predictor_var: list of predictor varaible names.
        target_var: target vaiable name.
        
        Performs parameter tuning for the specified 'param_set' on the given 'model'.
        prints the best parameter values and best score.
        
        Returns 'best_params_'        
        """
        
        grid_search = GridSearchCV(estimator = model, param_grid = param_set,n_jobs=-1, cv=5)
        grid_search.fit(train[predictor_var],train[target_var])
        
        print(grid_search.best_params_, grid_search.best_score_)
        
        return grid_search.best_params_    