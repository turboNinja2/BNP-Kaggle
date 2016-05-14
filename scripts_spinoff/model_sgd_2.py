import _import_data
import _cv_tools
import numpy as np
import csv
import sys
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV

max_categories = 35
arg_parser_index = 0
n_folds = 10
random_state_cv = 1 

while arg_parser_index < len(sys.argv) :
    if sys.argv[arg_parser_index] == '-max_categories':
        max_categories = int(sys.argv[arg_parser_index+1])
    if sys.argv[arg_parser_index] == '-random_state_cv':
        random_state_cv = int(sys.argv[arg_parser_index+1]) 
    arg_parser_index+=1

model_name = "sgd2_rscv"+str(random_state_cv)+'_'+str(max_categories)    
    
train, test, target, test_ids = _import_data.gen_data_5(max_categories)

model = SGDClassifier(loss="log",
                        penalty="elasticnet", 
                        n_jobs = -1,
                        n_iter = 100,
                        random_state = 123)
        
model = CalibratedClassifierCV(base_estimator=model, cv=5, method='isotonic')
 
score, predicted = _cv_tools.generic_cv(train,target, model, n_folds, random_state_cv)

model.fit(train,target)

np.savetxt("../gen_data/"+model_name+".train.csv", predicted, delimiter=",")

test_pred = model.predict_proba(test)[:,1]

np.savetxt("../gen_data/"+model_name+".test.csv", test_pred, delimiter=",")
