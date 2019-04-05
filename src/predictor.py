# -*- coding: utf-8 -*-
"""
Predictor.
@author: neeraj kesavan
"""

import pickle
import pandas as pd

class Predictor:
    """Class Predictor. To perform prediction on the given model and saves the prediction in kaggle format.
    
        Attributes:
            model_file_path: ath to the model to be loaded
            loaded_model: holds the loaded model
            target_var: target variable name.
            
        Methods:
            __init__(): Constructor. Initialize class attributes.
            predict(): Performs, prints  and returns prediction.
            save_predictions(): Performs prediction and saves it in kaggle format.
    """
    model_file_path = ''
    loaded_model = None
    target_var = ''
    
    def __init__(self, model_file_path='../model/model_final', target_var='SalePrice'):
        """Takes arguments 'model_file_path' and 'target_var'.
        model_file_path: path to the model tobe loaded. Default='../model/model_final'
        target_var: Target variable name. Default='SalePrice'
         
         Constructor. Initialize class attributes.
        """        
        self.model_file_path = model_file_path
        self.load_model()
        self.target_var = target_var
    
    def load_model(self):
        """Takes no argument.
        
        Loads model from the initialized path.
        """
        #Load the model from disk
        self.loaded_model = pickle.load(open(self.model_file_path, 'rb' ))
        
    def predict(self, test_data):        
        """Takes arguments 'model_file_path' and 'test_data'.
        model_file_path: filename from which model to be loaded.
        test_data: test data in which prediction to be made on.
                
        Prints  and returns the 'predictions'.
        """
        #Performs prediction.
        predictions = self.loaded_model.predict(test_data)
        
        print("\nPrediction")
        print(predictions)
        
        return predictions
    
    def save_predictions(self, test_data, output_file_path='../submission/predictions.csv'):
        """Takes arguments 'model_file_path' and 'test_data'.
        model_file_path: filename from which model to be loaded.
        test_data: test data in which prediction to be made on.
        
        Performs prediction and saves it in kaggle format.
        """

        #Performs prediction.
        predictions = self.loaded_model.predict(test_data.drop('Id', axis=1))
        
        #Saves to disk
        pred_file = pd.DataFrame()
        pred_file['Id'] = test_data['Id'].astype('int')
        pred_file[self.target_var] = predictions
        pred_file.to_csv(output_file_path, index=False)
        
        print("\nPredictions are saved  to disk..")