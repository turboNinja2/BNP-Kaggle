import _import_data
import _cv_tools
import numpy as np
import csv
import sys
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler

seed_value = 1
n_estimators = 10
max_samples = 10
arg_parser_index = 0
n_folds = 10
max_categories = 10
random_state_cv = 1

while arg_parser_index < len(sys.argv) :
    if sys.argv[arg_parser_index] == '-max_categories':
        max_categories = int(sys.argv[arg_parser_index+1])
    if sys.argv[arg_parser_index] == '-n_estimators':
        n_estimators = int(sys.argv[arg_parser_index+1])
    if sys.argv[arg_parser_index] == '-max_samples':
        max_samples = int(sys.argv[arg_parser_index+1])
    if sys.argv[arg_parser_index] == '-random_state_cv':
        random_state_cv = int(sys.argv[arg_parser_index+1])     
    arg_parser_index+=1


model_name = "svm1_" +str(max_categories)+"_"+str(max_samples)+"_"+str(n_estimators)+"_" +str(random_state_cv)   
    
if __name__ == '__main__':
    train, test, target, test_ids = _import_data.gen_data_4(max_categories)

    scaler = StandardScaler()       
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    
    model = BaggingClassifier(base_estimator=SVC(probability = True, kernel='rbf', random_state= seed_value),
                                  n_estimators=n_estimators,
                                  n_jobs=7,
                                  max_samples = max_samples)

    score, predicted = _cv_tools.generic_cv_np(train,target, model, n_folds, random_state_cv)

    model.fit(train,target)
    
    np.savetxt("../gen_data/"+model_name+".train.csv", predicted, delimiter=",")

    test_pred = model.predict_proba(test)[:,1]

    np.savetxt("../gen_data/"+model_name+".test.csv", test_pred, delimiter=",")
