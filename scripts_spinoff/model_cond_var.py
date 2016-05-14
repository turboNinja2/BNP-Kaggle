import _import_data
import _cv_tools
import numpy as np
import xgboost as xgb
import sys

arg_parser_index = 0
n_folds = 10
var_name = 'v22'
random_state_cv = 1

while arg_parser_index < len(sys.argv) :
    if sys.argv[arg_parser_index] == '-var_name':
        var_name = sys.argv[arg_parser_index+1]
    if sys.argv[arg_parser_index] == '-random_state_cv':
        random_state_cv = int(sys.argv[arg_parser_index+1])
    arg_parser_index+=1

    
train, test = _import_data.import_raw_data()



    
score, predicted = _cv_tools.generic_cv_reg(train,target, model, n_folds, random_state_cv)

model.fit(train,target)

np.savetxt("../gen_data/"+model_name+".train.csv", predicted, delimiter=",")
test_pred = model.predict(test)
np.savetxt("../gen_data/"+model_name+".test.csv", test_pred, delimiter=",")
