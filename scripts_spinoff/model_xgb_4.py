import _import_data
import _cv_tools
import numpy as np
import xgboost as xgb
import sys

arg_parser_index = 0
n_folds = 10

max_depth = 1
n_estimators = 10
max_categories=20
random_state_cv = 1

while arg_parser_index < len(sys.argv) :
    if sys.argv[arg_parser_index] == '-max_depth':
        max_depth = int(sys.argv[arg_parser_index+1])
    if sys.argv[arg_parser_index] == '-n_estimators':
        n_estimators = int(sys.argv[arg_parser_index+1])
    if sys.argv[arg_parser_index] == '-max_categories':
        max_categories = int(sys.argv[arg_parser_index+1])
    if sys.argv[arg_parser_index] == '-random_state_cv':
        random_state_cv = int(sys.argv[arg_parser_index+1])        
    arg_parser_index+=1

model_name = "xgb4_rscv"+str(random_state_cv)+'_'+str(max_depth)+'_'+str(n_estimators)+'_'+str(max_categories)
    
train, test, target, test_ids = _import_data.gen_data_4(max_categories)

model = xgb.XGBClassifier(n_estimators=n_estimators,
    nthread=-1,
    max_depth=max_depth,
    learning_rate=0.05,
    silent=True,
    subsample=0.8,
    colsample_bytree=0.8)
    
score, predicted = _cv_tools.generic_cv(train,target, model, n_folds, random_state_cv)

model.fit(train,target)

np.savetxt("../gen_data/"+model_name+".train.csv", predicted, delimiter=",")

test_pred = model.predict_proba(test)[:,1]

np.savetxt("../gen_data/"+model_name+".test.csv", test_pred, delimiter=",")
