import _import_data
import _cv_tools
import model_embedder
import numpy as np
import sys


from sklearn import ensemble

criterion = 'entropy'
arg_parser_index = 0
n_folds = 10
n_estimators = 10
random_state_cv = 1
max_depth = 50
min_leaf_split = 5

while arg_parser_index < len(sys.argv) :
    if sys.argv[arg_parser_index] == '-criterion':
        criterion = sys.argv[arg_parser_index+1]
    if sys.argv[arg_parser_index] == '-n_estimators':
        n_estimators = int(sys.argv[arg_parser_index+1])
    if sys.argv[arg_parser_index] == '-random_state_cv':
        random_state_cv = int(sys.argv[arg_parser_index+1])
    if sys.argv[arg_parser_index] == '-max_depth':
        max_depth = int(sys.argv[arg_parser_index+1]) 
    if sys.argv[arg_parser_index] == '-min_leaf_split':
        min_leaf_split = int(sys.argv[arg_parser_index+1])  
    arg_parser_index+=1

model_name = "erf5_rscv"+str(random_state_cv)+'_'+criterion+'_'+str(n_estimators)+'_'+str(max_depth)+'_'+str(min_leaf_split)
    
train, test, target, test_ids = _import_data.import_raw_data()

rf = ensemble.ExtraTreesClassifier(n_jobs=7, 
	n_estimators = n_estimators, 
	random_state = 11, 
	criterion = criterion,
    max_features = 50,
    min_samples_split= min_leaf_split,
    min_samples_leaf= min_leaf_split,
    max_depth= max_depth)

rf_embedded = model_embedder.ModelEmbedder(rf,10)
    
score, predicted = _cv_tools.generic_cv(train, target, rf_embedded, n_folds, random_state_cv)

rf_embedded.fit(train,target)

np.savetxt("../gen_data/"+model_name+".train.csv", predicted, delimiter=",")

test_pred = rf_embedded.predict_proba(test)[:,1]

np.savetxt("../gen_data/"+model_name+".test.csv", test_pred, delimiter=",")
