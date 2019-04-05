###Modules/Classes
Please start exploring from module main.py

All modules in this project are listed below:

* **data_processor.py** - Contains class DataProcessor to perform data cleansing, feature selection, feature extraction, outliers removal, unskewing and handles missing and categorical data. Class contains the following methods:
	
      	  `preprocess(self)`
      	  `populate_predictors_bytype(self)`
      	  `plot_data(self, train, X, Y, no_var=1)`
      	  `update_predictor_list(self, train)`
      	  `finalize_data(self, train, test)`
      	  `detect_outliers(self, train, threshold)`
      	  `get_features_on_variance(self, train_data, variance)`
      	  `unskew_data(self, train, test, columns, threshold)`
      	  `run(self)`
      	  	  
* **data_explorer.py** - Contains class DataExplorer to understand data with descriptive statistics and visualization. Class contains the following method.
	
		  `print_data_statistics(self, data)`
		  `visualize(self, data)`	

* **model_selector.py** - Contains class ModelSelector to perform model selection process and saves the final model for future predictions. Class contains the following methods:
	
	  	  `cross_validate(self, model, X, y, cv, n_jobs)` 
	  	  `finalize_and_save(self, model, filename, input_train, output_train))`
	  	  `get_feat_imp(self, model, train, predictor_vars, cv_folds, early_stopping_rounds, plot)`
	  	  `get_predictors_feat_imp(self, feature_imp, threshold)` 
	  	  `get_predictors(self, train)`
	  	  `run(self, train_data_path)`

* **ensemble_regressor.py** - Contains class EnsembleRegressor to ensemble multiple models. Class contains the following methods:
	
      	  `fit(self, train_X, train_y)`
      	  `predict(self, test_X)`
		  
* **parameter_tuner.py** - Contains class ParameterTuner  for hyperparameter tuning of models. Class contains the following method.
	
		  `tune_parameters(self, model, param_set, train, predictor_var, target_var)`

* **predictor.py** - Contains class Predictor to perform predictions on test data and saves predictions in kaggle format. Class contains the following methods:
	
	  	  `load_model(self)` 
	  	  `predict(self, test_data)`
	  	  `save_predictions(self, test_data, output_file_path):`

* **tuning_steps.py** - Contains iterative hyperparameter tuning process of models.
	  	  	
* **main.py** - To test and run the classes in this project.







