import _import_data
import _cv_tools
import numpy as np
import sys

from sklearn import ensemble

criterion = 'entropy'
arg_parser_index = 0
n_folds = 10
n_estimators = 10
random_state_cv = 1 

while arg_parser_index < len(sys.argv) :
    if sys.argv[arg_parser_index] == '-criterion':
        criterion = sys.argv[arg_parser_index+1]
    if sys.argv[arg_parser_index] == '-n_estimators':
        n_estimators = int(sys.argv[arg_parser_index+1])
    if sys.argv[arg_parser_index] == '-random_state_cv':
        random_state_cv = int(sys.argv[arg_parser_index+1])     
    arg_parser_index+=1

model_name = "rf_rscv"+str(random_state_cv)+'_'+criterion+'_'+str(n_estimators)

train, test, target, test_ids = _import_data.gen_data_1()

rf = ensemble.RandomForestClassifier(n_jobs=-1, 
	n_estimators = n_estimators, 
	random_state = 11, 
	criterion = criterion)
    
score, predicted = _cv_tools.generic_cv(train, target, rf, n_folds, random_state_cv)

rf.fit(train,target)

np.savetxt("../gen_data/"+model_name+".train.csv", predicted, delimiter=",")

test_pred = rf.predict_proba(test)[:,1]

np.savetxt("../gen_data/"+model_name+".test.csv", test_pred, delimiter=",")
