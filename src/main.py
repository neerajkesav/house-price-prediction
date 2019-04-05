# -*- coding: utf-8 -*-

from data_processor import DataProcessor
from model_selector import ModelSelector
from predictor import Predictor

import pandas as pd

##Data Processing
train_file_loc = '../resources/train.csv'
test_file_loc = '../resources/test.csv'
target_var ='SalePrice'
data_proc = DataProcessor(train_file_loc, test_file_loc, target_var ='SalePrice')
data_proc.run()

##Model Selection
select_model = ModelSelector()
train_file_loc = '../resources/train_final.csv'
select_model.run(train_file_loc)

##Prediction
pred = Predictor()
model_path = '../model/model_final'
test_file_loc = '../resources/test_final.csv'
test_data = pd.read_csv(test_file_loc)
pred.predict(test_data.drop('Id', axis=1))

#Saves prediction in kaggle format
pred.save_predictions(test_data)
