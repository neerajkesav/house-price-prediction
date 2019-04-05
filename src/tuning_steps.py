"""
Tuning Steps. This file consists of all hyperparameter tuning iterations.
@author: neeraj kesavan
"""
import pandas as pd
import numpy as np
import lightgbm as lgb

from xgboost.sklearn import XGBRegressor

from parameter_tuner import ParameterTuner
from model_selector import ModelSelector
from ensemble_regressor import EnsembleRegressor 
ms = ModelSelector()
dtrain = pd.read_csv('resources/train_final.csv')
predictor_vars = ms.get_predictors(dtrain)
target_var = 'SalePrice'

tuner = ParameterTuner()

xgboost = XGBRegressor(learning_rate=0.1,n_estimators=1000,
                                     max_depth=10, min_child_weight=5,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.2,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)


#Perform all Run 1 then try Run 2(Stops 2nd iteration if no change in values or no improvement)
#Run 1 & Run 2
tune_params = {'max_depth':np.arange(2,11,2).tolist(), 'min_child_weight':np.arange(1,6,2).tolist()}
best_params = tuner.tune_parameters(xgboost, tune_params, dtrain, predictor_vars, target_var)
#{'min_child_weight': 3, 'max_depth': 2} 0.9199922455535713
#{'min_child_weight': 5, 'max_depth': 2} 0.9212190642350695 #Run 2

tune_params = {'max_depth':[2,3,4], 'min_child_weight':[0,2,3,4]} #Run 1
tune_params = {'max_depth':[2,3,4], 'min_child_weight':[0,2,3,4,5,6]} #Run 2
best_params = tuner.tune_parameters(xgboost, tune_params, dtrain, predictor_vars, target_var)
#{'min_child_weight': 3, 'max_depth': 2} 0.9199922455535713 #Run 1
#{'min_child_weight': 2, 'max_depth': 2} 0.9223180931997698 #Run 2

tune_params = {'max_depth':[3,4,5], 'min_child_weight':[0,2,4]} #Run 1
best_params = tuner.tune_parameters(xgboost, tune_params, dtrain, predictor_vars, target_var)
#{'min_child_weight': 2, 'max_depth': 3} 0.9175698808322292 #Run 1

#Run 1 & Run 2
tune_params = {'max_depth':[2,3], 'min_child_weight':[2,3], 'learning_rate':np.arange(0.01,0.1,0.02).tolist()}
best_params = tuner.tune_parameters(xgboost, tune_params, dtrain, predictor_vars, target_var)
#{'min_child_weight': 2, 'learning_rate': 0.049999999999999996, 'max_depth': 2} 0.9207400878353837 #Run 1
#{'min_child_weight': 3, 'learning_rate': 0.049999999999999996, 'max_depth': 2} 0.9212521874412405 #Run 2

#Run 1 & Run 2
tune_params = {'max_depth':[2,3], 'min_child_weight':[2,3], 'learning_rate':np.arange(0.02,0.1,0.02).tolist()}
best_params = tuner.tune_parameters(xgboost, tune_params, dtrain, predictor_vars, target_var)
#{'min_child_weight': 2, 'learning_rate': 0.06, 'max_depth': 2} 0.9211704818734688 #Run 1
#{'min_child_weight': 2, 'learning_rate': 0.06, 'max_depth': 2} 0.9223180931997698 #Run 2
 
#Run 1
tune_params = {'max_depth':[2,3], 'min_child_weight':[2,3], 'learning_rate':np.arange(0.001,0.01,0.002).tolist()}
best_params = tuner.tune_parameters(xgboost, tune_params, dtrain, predictor_vars, target_var)
#{'min_child_weight': 3, 'learning_rate': 0.009000000000000001, 'max_depth': 3} 0.9132484718984368 #Run 1

##Run 1
xgboost = XGBRegressor(learning_rate=0.06,n_estimators=1000,
                                     max_depth=2, min_child_weight=2,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.2,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)

#Run 1
tune_params = {'gamma': [i/10.0 for i in range(0,5)]}
best_params = tuner.tune_parameters(xgboost, tune_params, dtrain, predictor_vars, target_var)
#{'gamma': 0.0} 0.9211704818734688 #Run 1

#Run 1 & 2
tune_params = {'subsample':[i/10.0 for i in range(2,10)], 'colsample_bytree':[i/10.0 for i in range(2,10)]}
best_params = tuner.tune_parameters(xgboost, tune_params, dtrain, predictor_vars, target_var)
#{'colsample_bytree': 0.2, 'subsample': 0.4} 0.9217078503219566 #Run 1
#{'colsample_bytree': 0.2, 'subsample': 0.4} 0.9223180931997698 #Run 2

#Run 1 & 2
tune_params = {'subsample':[i/100.0 for i in range(35,55,5)], 'colsample_bytree':[i/100.0 for i in range(15,35,5)]}
best_params = tuner.tune_parameters(xgboost, tune_params, dtrain, predictor_vars, target_var)
#{'colsample_bytree': 0.2, 'subsample': 0.4} 0.9217078503219566 #Run 1
#{'colsample_bytree': 0.2, 'subsample': 0.4} 0.9223180931997698 #Run 2

#Run 1
xgboost = XGBRegressor(learning_rate=0.06,n_estimators=1000,
                                     max_depth=2, min_child_weight=2,
                                     gamma=0, subsample=0.4,
                                     colsample_bytree=0.2,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)

#Run 1
tune_params = {'reg_alpha':[0.00001, 0.0001, 0.001, .01, 0.1, 1, 10, 100]}
best_params = tuner.tune_parameters(xgboost, tune_params, dtrain, predictor_vars, target_var)
#{'reg_alpha': 100} 0.9220422048979041 #Run 1

tune_params = {'reg_alpha':[50,75,100,125]}
best_params = tuner.tune_parameters(xgboost, tune_params, dtrain, predictor_vars, target_var)
#{'reg_alpha': 75} 0.9223180556958955 #Run 1

tune_params = {'reg_alpha':[65,70,75,80,85]}
best_params = tuner.tune_parameters(xgboost, tune_params, dtrain, predictor_vars, target_var)
#{'reg_alpha': 75} 0.9223180556958955 #Run 1

tune_params = {'reg_alpha':[71,73,75,77,79]}
best_params = tuner.tune_parameters(xgboost, tune_params, dtrain, predictor_vars, target_var)
#{'reg_alpha': 77} 0.9223180931997698 #Run 1

#Run 1
xgboost = XGBRegressor(learning_rate=0.06,n_estimators=1000,
                                     max_depth=2, min_child_weight=2,
                                     gamma=0, subsample=0.4,
                                     colsample_bytree=0.2,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=77)

#CV Score is: 0.9223215284352874

############################################################################

"""MODEL 2"""
tuner = ParameterTuner()

xgboost2 = XGBRegressor(learning_rate=0.1,n_estimators=1500,
                                     max_depth=10, min_child_weight=5,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.2,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=99,
                                     reg_alpha=0.00006)


#Run 1 & 2
tune_params = {'max_depth':np.arange(2,11,2).tolist(), 'min_child_weight':np.arange(1,6,2).tolist()}
best_params = tuner.tune_parameters(xgboost2, tune_params, dtrain, predictor_vars, target_var)
#{'min_child_weight': 1, 'max_depth': 2} 0.919599754495862 #Run 1
#{'min_child_weight': 1, 'max_depth': 2} 0.921733593442868 #Run 2

#Run 1 & 2
tune_params = {'max_depth':[2, 3, 5], 'min_child_weight':[0,1,2,4]}
best_params = tuner.tune_parameters(xgboost2, tune_params, dtrain, predictor_vars, target_var)
#{'min_child_weight': 0, 'max_depth': 2} 0.919599754495862 #Run 1
#{'min_child_weight': 0, 'max_depth': 2} 0.921733593442868 #Run 2

#Run 1
xgboost2 = XGBRegressor(learning_rate=0.1,n_estimators=1500,
                                     max_depth=2, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.2,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=99,
                                     reg_alpha=0.00006)

#Run 1 & 2
tune_params = {'learning_rate':np.arange(0.01,0.1,0.02).tolist()}
best_params = tuner.tune_parameters(xgboost2, tune_params, dtrain, predictor_vars, target_var)
#{'learning_rate': 0.03} 0.9206925577830832 #Run 1
#{'learning_rate': 0.03} 0.9206925744939465 #Run 2

#Run 1 & 2
tune_params = {'learning_rate':np.arange(0.02,0.1,0.02).tolist()}
best_params = tuner.tune_parameters(xgboost2, tune_params, dtrain, predictor_vars, target_var)
#{'learning_rate': 0.04} 0.9217335622311865 #Run 1
#{'learning_rate': 0.04} 0.921733593442868 #Run 2

#Run 1
xgboost2 = XGBRegressor(learning_rate=0.04,n_estimators=1500,
                                     max_depth=2, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.2,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=99,
                                     reg_alpha=0.00006)

#Run 1
tune_params = {'gamma': [i/10.0 for i in range(0,5)]}
best_params = tuner.tune_parameters(xgboost2, tune_params, dtrain, predictor_vars, target_var)
#{'gamma': 0.0} 0.9217335622311865 #Run 1

#Run 1 & 2
tune_params = {'subsample':[i/10.0 for i in range(2,10)], 'colsample_bytree':[i/10.0 for i in range(2,10)]}
best_params = tuner.tune_parameters(xgboost2, tune_params, dtrain, predictor_vars, target_var)
#{'colsample_bytree': 0.2, 'subsample': 0.7} 0.9217335622311865 #Run 1
#{'colsample_bytree': 0.2, 'subsample': 0.7} 0.921733593442868 #Run 2

#Run 1
tune_params = {'subsample':[i/100.0 for i in range(55,85,5)], 'colsample_bytree':[i/100.0 for i in range(15,35,5)]}
best_params = tuner.tune_parameters(xgboost2, tune_params, dtrain, predictor_vars, target_var)
#{'colsample_bytree': 0.2, 'subsample': 0.7} 0.9217335622311865 #Run 1

#Run 1
tune_params = {'reg_alpha':[0.00001, 0.0001, 0.001, .01, 0.1, 1, 10, 100]}
best_params = tuner.tune_parameters(xgboost2, tune_params, dtrain, predictor_vars, target_var)
#{'reg_alpha': 1} 0.9217335771231051

#Run 1
tune_params = {'reg_alpha':[0.1,0.5,1,1.5,2,2.5]}
best_params = tuner.tune_parameters(xgboost2, tune_params, dtrain, predictor_vars, target_var)
#{'reg_alpha': 1.5} 0.9217335905788014

#Run 1
tune_params = {'reg_alpha':[1.3,1.5,1.7]}
best_params = tuner.tune_parameters(xgboost2, tune_params, dtrain, predictor_vars, target_var)
#{'reg_alpha': 1.7} 0.921733593442868

#Run 1
xgboost2 = XGBRegressor(learning_rate=0.04,n_estimators=1500,
                                     max_depth=2, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.2,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=99,
                                     reg_alpha=1.7)
#CV Score is: 0.9217399806650176
###############
""" MODEL 3"""
xgboost3 = XGBRegressor(learning_rate=0.1,n_estimators=1200,
                                     max_depth=10, min_child_weight=5,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.2,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=585,
                                     reg_alpha=0.00006)

#Run 1 & 2
tune_params = {'max_depth':np.arange(2,11,2).tolist(), 'min_child_weight':np.arange(1,6,2).tolist()}
best_params = tuner.tune_parameters(xgboost3, tune_params, dtrain, predictor_vars, target_var)
#{'min_child_weight': 1, 'max_depth': 2} 0.9137039370777823 #Run 1
#{'min_child_weight': 5, 'max_depth': 2} 0.9210777919074208 #Run 2

tune_params = {'max_depth':[2, 3, 5], 'min_child_weight':[0,1,2,4]} #Run 1
tune_params = {'max_depth':[2, 3, 5], 'min_child_weight':[0,2,4,6]} #Run 2
best_params = tuner.tune_parameters(xgboost3, tune_params, dtrain, predictor_vars, target_var)
#{'min_child_weight': 2, 'max_depth': 3} 0.9150130657543103 #Run 1
#{'min_child_weight': 6, 'max_depth': 3} 0.9225494994852237 #Run 2

tune_params = {'max_depth':[2, 3], 'min_child_weight':[2,6,8,10]} #Run 2
best_params = tuner.tune_parameters(xgboost3, tune_params, dtrain, predictor_vars, target_var)
#{'min_child_weight': 6, 'max_depth': 3} 0.9225494994852237 #Run 2

#Run 1
xgboost3 = XGBRegressor(learning_rate=0.1,n_estimators=1200,
                                     max_depth=3, min_child_weight=2,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.2,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=585,
                                     reg_alpha=0.00006)
#Update the parameters from Run 2
#Run 2
xgboost3 = XGBRegressor(learning_rate=0.02,n_estimators=1200,
                                     max_depth=3, min_child_weight=6,
                                     gamma=0, subsample=0.55,
                                     colsample_bytree=0.25,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=585,
                                     reg_alpha=0.5)

#Run 1 & 2                                
tune_params = {'learning_rate':np.arange(0.01,0.1,0.02).tolist()}
best_params = tuner.tune_parameters(xgboost3, tune_params, dtrain, predictor_vars, target_var)
#{'learning_rate': 0.03} 0.9212673017325689 #Run 1
#{'learning_rate': 0.03} 0.9207570932636977 #Run 2

#Run 1 & 2
tune_params = {'learning_rate':np.arange(0.02,0.1,0.02).tolist()}
best_params = tuner.tune_parameters(xgboost3, tune_params, dtrain, predictor_vars, target_var)
#{'learning_rate': 0.02} 0.9219669648293832 #Run 1
#{'learning_rate': 0.02} 0.9225494994852237 #Run 2

#Run 1
xgboost3 = XGBRegressor(learning_rate=0.02,n_estimators=1200,
                                     max_depth=3, min_child_weight=2,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.2,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=585,
                                     reg_alpha=0.00006)
#Run 1
tune_params = {'gamma': [i/10.0 for i in range(0,5)]}
best_params = tuner.tune_parameters(xgboost3, tune_params, dtrain, predictor_vars, target_var)
#{'gamma': 0.0} 0.9219669648293832 #Run 1

#Run 1 & 2
tune_params = {'subsample':[i/10.0 for i in range(2,10)], 'colsample_bytree':[i/10.0 for i in range(2,10)]}
best_params = tuner.tune_parameters(xgboost3, tune_params, dtrain, predictor_vars, target_var)
#{'colsample_bytree': 0.2, 'subsample': 0.7} 0.9219669648293832 #Run 1
#{'colsample_bytree': 0.2, 'subsample': 0.7} 0.922257826069896 #Run 2

#Run 1 & 2
tune_params = {'subsample':[i/100.0 for i in range(55,85,5)], 'colsample_bytree':[i/100.0 for i in range(15,35,5)]}
best_params = tuner.tune_parameters(xgboost3, tune_params, dtrain, predictor_vars, target_var)
#{'colsample_bytree': 0.25, 'subsample': 0.55} 0.9220519354226215 #Run 1
#{'colsample_bytree': 0.2, 'subsample': 0.65} 0.9228026158409246 #Run 2

#Run 1
xgboost3 = XGBRegressor(learning_rate=0.02,n_estimators=1200,
                                     max_depth=3, min_child_weight=2,
                                     gamma=0, subsample=0.55,
                                     colsample_bytree=0.25,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=585,
                                     reg_alpha=0.00006)
#Update parametrs from Run 2
#Run 2
xgboost3 = XGBRegressor(learning_rate=0.02,n_estimators=1200,
                                     max_depth=3, min_child_weight=2,
                                     gamma=0, subsample=0.65,
                                     colsample_bytree=0.2,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=585,
                                     reg_alpha=0.5)

#Run 1 & 2
tune_params = {'reg_alpha':[0.00001, 0.0001, 0.001, .01, 0.1, 1, 10, 100]}
best_params = tuner.tune_parameters(xgboost3, tune_params, dtrain, predictor_vars, target_var)
#{'reg_alpha': 1} 0.9221243691038352 #Run 1
#{'reg_alpha': 100} 0.9215502958996129 #Run 2

#Run 1 & 2
tune_params = {'reg_alpha':[0.1,0.5,1,1.5,2,2.5]}
best_params = tuner.tune_parameters(xgboost3, tune_params, dtrain, predictor_vars, target_var)
#{'reg_alpha': 0.5} 0.9221243772549602 #Run 1
#{'reg_alpha': 0.1} 0.9215488443932763 #Run 2

#Run 1 & 2
tune_params = {'reg_alpha':[0.25,0.5,0.75]}
best_params = tuner.tune_parameters(xgboost3, tune_params, dtrain, predictor_vars, target_var)
#{'reg_alpha': 0.5} 0.9221243772549602 #Run 1
#{'reg_alpha': 0.25} 0.9215488342609 #Run 2

#Run 2
tune_params = {'reg_alpha':[0.00006,150,125,175]}
tune_params = {'reg_alpha':[175,350,500,800]}
tune_params = {'reg_alpha':[800,2000,1500]}
tune_params = {'reg_alpha':[1500,5000,7000]}
tune_params = {'reg_alpha':[4500,5000,5500]}
best_params = tuner.tune_parameters(xgboost3, tune_params, dtrain, predictor_vars, target_var)
#{'reg_alpha': 5000} 0.922545553458107 #Run 2

#Run 1
xgboost3 = XGBRegressor(learning_rate=0.02,n_estimators=1200,
                                     max_depth=3, min_child_weight=2,
                                     gamma=0, subsample=0.55,
                                     colsample_bytree=0.25,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=585,
                                     reg_alpha=5000)
#CV Score is: 0.9221258949104317   

#Update the parameters from Run 2                                     
#Run 2
xgboost3 = XGBRegressor(learning_rate=0.02,n_estimators=1200,
                                     max_depth=3, min_child_weight=2,
                                     gamma=0, subsample=0.65,
                                     colsample_bytree=0.2,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=585,
                                     reg_alpha=5000)                                     
#CV Score is: 0.9225488378885822
######################################################################

""" LIGHT GBM """
 
lightgbm = lgb.LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       learning_rate=0.1, 
                                       n_estimators=3000,
                                       max_bin=200, 
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=9,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=9,
                                       data_random_seed=9,
                                       verbose=-1)

#Run 1 & 2
tune_params = {'num_leaves':[i for i in range(2,11,2)], 'min_data_in_leaf':[i for i in range(5,21,5)]}
best_params = tuner.tune_parameters(lightgbm, tune_params, dtrain, predictor_vars, target_var)
#{'min_data_in_leaf': 10, 'num_leaves': 4} 0.912042969681592 #Run 1
#{'min_data_in_leaf': 5, 'num_leaves': 4} 0.9142370738944078 #Run 2

#Run 1 & 2
tune_params = {'num_leaves':[i for i in range(3,11,2)], 'min_data_in_leaf':[i for i in range(3,10,2)]}
best_params = tuner.tune_parameters(lightgbm, tune_params, dtrain, predictor_vars, target_var)
#{'min_data_in_leaf': 8, 'num_leaves': 3} 0.9153736468226764 #Run 1
#{'min_data_in_leaf': 3, 'num_leaves': 9} 0.9153869460714626 #Run 2

#Run 1 & 2
tune_params = {'num_leaves':[3,4,5,8,9,10], 'min_data_in_leaf':[3,5,8,10]}
best_params = tuner.tune_parameters(lightgbm, tune_params, dtrain, predictor_vars, target_var)
#{'min_data_in_leaf': 8, 'num_leaves': 3} 0.9153736468226764 #Run 1
#{'min_data_in_leaf': 3, 'num_leaves': 9} 0.9153869460714626 #Run 2
#Run 1 & 2
tune_params = {'num_leaves':[4,9], 'min_data_in_leaf':[3,5], 'learning_rate':np.arange(0.01,0.1,0.02).tolist()}
best_params = tuner.tune_parameters(lightgbm, tune_params, dtrain, predictor_vars, target_var)
#{'min_data_in_leaf': 8, 'learning_rate': 0.01, 'num_leaves': 4} 0.9213444768447892 #Run 1
#{'min_data_in_leaf': 5, 'learning_rate': 0.01, 'num_leaves': 4} 0.9220405884198094 #Run 2

#Run 1
tune_params = {'num_leaves':[4,9], 'min_data_in_leaf':[3,5], 'learning_rate':[0.001, 0.004, 0.008, 0.01, 0.015, 0.02]}
best_params = tuner.tune_parameters(lightgbm, tune_params, dtrain, predictor_vars, target_var)
#{'min_data_in_leaf': 5, 'learning_rate': 0.02, 'num_leaves': 4} 0.9225306574199844 #Run 1

#Run 1
tune_params = {'num_leaves':[4], 'min_data_in_leaf':[5], 'learning_rate':[0.02,0.04,0.06]}
best_params = tuner.tune_parameters(lightgbm, tune_params, dtrain, predictor_vars, target_var)
#{'min_data_in_leaf': 5, 'learning_rate': 0.02, 'num_leaves': 4} 0.9225306574199844 #Run 1

#Run 1
lightgbm = lgb.LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       min_data_in_leaf=5,
                                       learning_rate=0.02, 
                                       n_estimators=3000,
                                       max_bin=200, 
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=9,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=9,
                                       data_random_seed=9,
                                       verbose=-1)

#Run 1
tune_params = {'bagging_fraction':[i/10.0 for i in range(1,10,2)], 'bagging_freq':[i for i in range(5,21,5)]}
best_params = tuner.tune_parameters(lightgbm, tune_params, dtrain, predictor_vars, target_var)
#{'bagging_freq': 10, 'bagging_fraction': 0.7} 0.9226565557681654 #Run 1

#Run 1
tune_params = {'bagging_fraction':[i/100.0 for i in range(55,90,5)], 'bagging_freq':[i for i in range(8,14,2)]}
best_params = tuner.tune_parameters(lightgbm, tune_params, dtrain, predictor_vars, target_var)
#{'bagging_freq': 10, 'bagging_fraction': 0.85} 0.9232429761926936 #Run 1

#Run 1
lightgbm = lgb.LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       min_data_in_leaf=5,
                                       learning_rate=0.02, 
                                       n_estimators=3000,
                                       max_bin=200, 
                                       bagging_fraction=0.85,
                                       bagging_freq=10, 
                                       bagging_seed=9,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=9,
                                       data_random_seed=9,
                                       verbose=-1)

#Run 1
tune_params = {'feature_fraction':[i/10.0 for i in range(1,10,2)]}
best_params = tuner.tune_parameters(lightgbm, tune_params, dtrain, predictor_vars, target_var)
#{'feature_fraction': 0.3} 0.921614667349321 #Run 1

#Run 1
tune_params = {'feature_fraction':[0.2,0.3,0.4,0.6]}
best_params = tuner.tune_parameters(lightgbm, tune_params, dtrain, predictor_vars, target_var)
#{'feature_fraction': 0.2} 0.9232429761926936 #Run 1

#Run 1
tune_params = {'feature_fraction':[0.15,0.2,0.25,0.8,1]}
best_params = tuner.tune_parameters(lightgbm, tune_params, dtrain, predictor_vars, target_var)
#{'feature_fraction': 0.2} 0.9232429761926936 #Run 1

#Run 1
tune_params = {'max_bin': [i for i in range(200,400,55)]}
best_params = tuner.tune_parameters(lightgbm, tune_params, dtrain, predictor_vars, target_var)
#{'max_bin': 365} 0.9243154645959906 #Run 1

#Run 1
tune_params = {'max_bin': [150,200,340,365,400,445]}
best_params = tuner.tune_parameters(lightgbm, tune_params, dtrain, predictor_vars, target_var)
#{'max_bin': 365} 0.9243154645959906 #Run 1

#Run 1
tune_params = {'max_bin': [350,355,365,365,370]}
best_params = tuner.tune_parameters(lightgbm, tune_params, dtrain, predictor_vars, target_var)
#{'max_bin': 350} 0.9247014594383771 #Run 1

#Run 1
tune_params = {'max_bin': [330,335,340,345,350]}
best_params = tuner.tune_parameters(lightgbm, tune_params, dtrain, predictor_vars, target_var)
#{'max_bin': 330} 0.9248008989902761 #Run 1

#Run 1
tune_params = {'max_bin': [250,275,300,315,320,325,330,335]}
best_params = tuner.tune_parameters(lightgbm, tune_params, dtrain, predictor_vars, target_var)
#{'max_bin': 320} 0.9249069107342514 #Run 1

#Run 1
lightgbm = lgb.LGBMRegressor(objective='regression', 
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
                                       verbose=-1)

#Run 1
tune_params = {'reg_alpha': [i/10.0 for i in range(0,10,2)], 'reg_lambda':[i/10.0 for i in range(0,10,2)] }
best_params = tuner.tune_parameters(lightgbm, tune_params, dtrain, predictor_vars, target_var)
#{'reg_lambda': 0.0, 'reg_alpha': 0.6} 0.9250544786935161 

#Run 1
tune_params = {'reg_alpha': [i/10.0 for i in range(1,10,2)], 'reg_lambda':[0.0,0.1,0.3] }
best_params = tuner.tune_parameters(lightgbm, tune_params, dtrain, predictor_vars, target_var)
#{'reg_lambda': 0.3, 'reg_alpha': 0.5} 0.9251308877415608

#Run 1
tune_params = {'reg_alpha': [0.5,0.55,0.6,0.65], 'reg_lambda':[0.0,0.3,0.5,0.7,0.9] }
best_params = tuner.tune_parameters(lightgbm, tune_params, dtrain, predictor_vars, target_var)
#{'reg_lambda': 0.3, 'reg_alpha': 0.55} 0.9251708951648705

#Run 1
tune_params = {'reg_alpha':[0.00001, 0.0001, 0.001, .01, 0.55, 10, 100], 'reg_lambda': [0.3]}
best_params = tuner.tune_parameters(lightgbm, tune_params, dtrain, predictor_vars, target_var)
#{'reg_lambda': 0.3, 'reg_alpha': 0.55} 0.9251708951648705

#Run 1
lightgbm = lgb.LGBMRegressor(objective='regression', 
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

#CV Score is: 0.9230228511066916

#######################################################################

""" MODEL 5 """

lightgbm2 = lgb.LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       min_data_in_leaf=5,
                                       learning_rate=0.02, 
                                       n_estimators=4000,
                                       max_bin=320, 
                                       bagging_fraction=0.85,
                                       bagging_freq=10, 
                                       bagging_seed=24,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=24,
                                       data_random_seed=24,
                                       reg_alpha=0.55,
                                       reg_lambda=0.3,
                                       verbose=-1) 

#Run 1
tune_params = {'num_leaves':[3,4,5], 'min_data_in_leaf':[4,5,6,10]}
best_params = tuner.tune_parameters(lightgbm2, tune_params, dtrain, predictor_vars, target_var)
#{'min_data_in_leaf': 5, 'num_leaves': 3} 0.9227920858348057

#Run 1
tune_params = {'num_leaves':[3,4], 'min_data_in_leaf':[3,5], 'learning_rate':np.arange(0.01,0.1,0.02).tolist()}
best_params = tuner.tune_parameters(lightgbm2, tune_params, dtrain, predictor_vars, target_var)
#{'min_data_in_leaf': 3, 'learning_rate': 0.01, 'num_leaves': 4} 0.9228281827223667

#Run 1
tune_params = {'num_leaves':[3,4], 'min_data_in_leaf':[3,5], 'learning_rate':[0.001, 0.004, 0.008, 0.01, 0.015, 0.02]}
best_params = tuner.tune_parameters(lightgbm2, tune_params, dtrain, predictor_vars, target_var)
#{'min_data_in_leaf': 3, 'learning_rate': 0.01, 'num_leaves': 4} 0.9228281827223667

#Run 1
lightgbm2 = lgb.LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       min_data_in_leaf=3,
                                       learning_rate=0.01, 
                                       n_estimators=4000,
                                       max_bin=320, 
                                       bagging_fraction=0.85,
                                       bagging_freq=10, 
                                       bagging_seed=24,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=24,
                                       data_random_seed=24,
                                       reg_alpha=0.55,
                                       reg_lambda=0.3,
                                       verbose=-1)

#Run 1 
tune_params = {'bagging_fraction':[i/10.0 for i in range(1,10,2)], 'bagging_freq':[i for i in range(5,21,5)]}
best_params = tuner.tune_parameters(lightgbm2, tune_params, dtrain, predictor_vars, target_var)
#{'bagging_freq': 10, 'bagging_fraction': 0.5} 0.9241586885847276

#Run 1
tune_params = {'bagging_fraction':[i/100.0 for i in range(35,75,5)], 'bagging_freq':[i for i in range(8,14,2)]}
best_params = tuner.tune_parameters(lightgbm2, tune_params, dtrain, predictor_vars, target_var)
#{'bagging_freq': 10, 'bagging_fraction': 0.5} 0.9241586885847276 

#Run 1
lightgbm2 = lgb.LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       min_data_in_leaf=3,
                                       learning_rate=0.01, 
                                       n_estimators=4000,
                                       max_bin=320, 
                                       bagging_fraction=0.5,
                                       bagging_freq=10, 
                                       bagging_seed=24,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=24,
                                       data_random_seed=24,
                                       reg_alpha=0.55,
                                       reg_lambda=0.3,
                                       verbose=-1)

#Run 1
tune_params = {'feature_fraction':[i/10.0 for i in range(1,10,2)]}
best_params = tuner.tune_parameters(lightgbm2, tune_params, dtrain, predictor_vars, target_var)
#{'feature_fraction': 0.3} 0.9227769089068331

#Run 1
tune_params = {'feature_fraction':[0.2,0.3,0.4,0.6]}
best_params = tuner.tune_parameters(lightgbm2, tune_params, dtrain, predictor_vars, target_var)
#{'feature_fraction': 0.2} 0.9241586885847276

#Run 1
tune_params = {'max_bin': [200,250,300,320,350]}
best_params = tuner.tune_parameters(lightgbm2, tune_params, dtrain, predictor_vars, target_var)
#{'max_bin': 320} 0.9241586885847276

#Run 1
tune_params = {'max_bin': [150,310,315,320,325,330]}
best_params = tuner.tune_parameters(lightgbm2, tune_params, dtrain, predictor_vars, target_var)
#{'max_bin': 310} 0.9242037757557104

#Run 1
tune_params = {'max_bin': [295,300,305,310]}
best_params = tuner.tune_parameters(lightgbm2, tune_params, dtrain, predictor_vars, target_var)
#{'max_bin': 295} 0.9244484530460247

#Run 1
tune_params = {'max_bin': [265,275,280,285,290,295]}
best_params = tuner.tune_parameters(lightgbm2, tune_params, dtrain, predictor_vars, target_var)
#{'max_bin': 295} 0.9244484530460247

#Run 1
lightgbm2 = lgb.LGBMRegressor(objective='regression', 
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
                                       reg_alpha=0.55,
                                       reg_lambda=0.3,
                                       verbose=-1)

#Run 1
tune_params = {'reg_alpha': [i/10.0 for i in range(0,10,2)], 'reg_lambda':[i/10.0 for i in range(0,10,2)] }
best_params = tuner.tune_parameters(lightgbm2, tune_params, dtrain, predictor_vars, target_var)
#{'reg_lambda': 0.6, 'reg_alpha': 0.8} 0.924513096718245

#Run 1
tune_params = {'reg_alpha': [i/10.0 for i in range(1,10,2)], 'reg_lambda':[i/10.0 for i in range(1,10,2)] }
best_params = tuner.tune_parameters(lightgbm2, tune_params, dtrain, predictor_vars, target_var)
#{'reg_lambda': 0.7, 'reg_alpha': 0.3} 0.9244655027804274

#Run 1
tune_params = {'reg_alpha': [0.3,0.25,0.35,0.75,0.80,85], 'reg_lambda':[0.55,0.6,0.65,0.7,0.75] }
best_params = tuner.tune_parameters(lightgbm2, tune_params, dtrain, predictor_vars, target_var)
#{'reg_lambda': 0.7, 'reg_alpha': 0.8} 0.9245349605422362

#Run 1
tune_params = {'reg_alpha':[0.00001, 0.0001, 0.001, .01, 0.8, 10, 100], 'reg_lambda': [0.7]}
best_params = tuner.tune_parameters(lightgbm2, tune_params, dtrain, predictor_vars, target_var)
#{'reg_lambda': 0.7, 'reg_alpha': 10} 0.9245934436927781

#Run 1
tune_params = {'reg_alpha':[0.8,10], 'reg_lambda': [0.00001, 0.0001, 0.001, .01, 0.7, 10, 100]}
best_params = tuner.tune_parameters(lightgbm2, tune_params, dtrain, predictor_vars, target_var)
#{'reg_lambda': 0.7, 'reg_alpha': 10} 0.9245934436927781

#Run 1
tune_params = {'reg_alpha':[0.8,5,10,15,20,50], 'reg_lambda': [0.7]}
best_params = tuner.tune_parameters(lightgbm2, tune_params, dtrain, predictor_vars, target_var)
#{'reg_lambda': 0.7, 'reg_alpha': 10} 0.9245934436927781

#Run 1
tune_params = {'reg_alpha':[8,10,12,0.8], 'reg_lambda': [0.7]}
best_params = tuner.tune_parameters(lightgbm2, tune_params, dtrain, predictor_vars, target_var)
#{'reg_lambda': 0.7, 'reg_alpha': 10} 0.9245934436927781

#Run 1
lightgbm2 = lgb.LGBMRegressor(objective='regression', 
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

#CV Score is: 0.9243765697929301 
 
                      
#######################################################################  

""" ENSEMBLING """    
                                                                 
xgb_ens = EnsembleRegressor([xgboost,xgboost2, xgboost3])
#CV Score is: 0.9246359450211432                                     
xgb_ens = EnsembleRegressor([xgboost,xgboost2, xgboost3, lightgbm])
#CV Score is: 0.9249748684043093                                   
xgb_ens = EnsembleRegressor([xgboost,xgboost2, xgboost3, lightgbm, lightgbm2])
#CV Score is: 0.92528287952747       
#CV Score is: 0.9253181909342896                             
###################################################################### 
                                     
""" CROSS VALIDATION""" 

ms =ModelSelector()
ms.cross_validate(cv=5,model=xgb_ens,X=dtrain.drop(['SalePrice'], axis=1)[predictor_vars], y=dtrain['SalePrice'], n_jobs = 1)
#CV Score is: 0.92528287952747 all predictor variables
                                    
#Using feature importance to check for improvement.
ft_imp = ms.get_feat_imp(model=xgboost3, train=dtrain, predictor_vars=predictor_vars, target_var=target_var,cv_folds=5, early_stopping_rounds=50, plot=False)

colList = ms.get_predictors_feat_imp(feature_imp=ft_imp, threshold=5)
ms.cross_validate(cv=5,model=xgb_ens,X=dtrain.drop(['SalePrice'], axis=1)[colList], y=dtrain['SalePrice'], n_jobs = 1)
#CV Score is: 0.9237782931789443 ft_imp>0
#CV Score is: 0.9245306746519069 ft_imp>5
#CV Score is: 0.9241550112166916 ft_imp>3
#CV Score is: 0.9240815634686982 ft_imp>6
#CV Score is: 0.9241180904319265 ft_imp>4
# No improvement in using feature importance in this case