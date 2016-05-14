import _import_data
import _cv_tools
import numpy as np
import xgboost as xgb
import sys
from OneHotNaiveBayes import OneHotNaiveBayes

arg_parser_index = 0
n_folds = 5
var_name1 = 'v22'
var_name2 = 'v22'
random_state_cv = 10

while arg_parser_index < len(sys.argv) :
    if sys.argv[arg_parser_index] == '-var_name1':
        var_name1 = sys.argv[arg_parser_index+1]
    if sys.argv[arg_parser_index] == '-var_name2':
        var_name2 = sys.argv[arg_parser_index+1]
    if sys.argv[arg_parser_index] == '-random_state_cv':
        random_state_cv = int(sys.argv[arg_parser_index+1])
    arg_parser_index+=1

model_name = "ohnb_"+str(random_state_cv)+'_'+var_name1+'_'+var_name2

train, test, target, test_ids = _import_data.import_raw_data()

train = _import_data.interactions(train,var_name1,var_name2)
test = _import_data.interactions(test,var_name1,var_name2)

model = OneHotNaiveBayes(var_name1+'_'+var_name2,'target')
    
score, predicted = _cv_tools.generic_cv_reg(train, target, model, n_folds, random_state_cv)

model.fit(train,target)

np.savetxt("../gen_data/"+model_name+".train.csv", predicted, delimiter=",")

test_pred = model.predict(test)

np.savetxt("../gen_data/"+model_name+".test.csv", test_pred, delimiter=",")
