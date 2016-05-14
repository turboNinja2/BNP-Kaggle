import _import_data
import _cv_tools
import numpy as np
import pandas as pd
import sys

from sklearn import ensemble

criterion = 'entropy'
arg_parser_index = 0
n_folds = 10
n_estimators = 10
random_state_cv = 1
max_depth = 50
min_leaf_split = 5
max_features = 50

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
    if sys.argv[arg_parser_index] == '-max_features':
        max_features = int(sys.argv[arg_parser_index+1])  
    arg_parser_index+=1

model_name = "serf1_rscv"+str(random_state_cv)+'_'+criterion+'_'+str(n_estimators)+'_'+str(max_depth)+'_'+str(min_leaf_split) + '_' +str(max_features)
    
print('Load data...')
train = pd.read_csv("../input/train.csv")
target = train['target']
train = train.drop(['ID','target','v8','v23','v25','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)

test = pd.read_csv("../input/test.csv")
id_test = test['ID'].values
test = test.drop(['ID','v8','v23','v25','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)

print('Clearing...')
for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
    if train_series.dtype == 'O':
        #for objects: factorize
        train[train_name], tmp_indexer = pd.factorize(train[train_name])
        test[test_name] = tmp_indexer.get_indexer(test[test_name])
        #but now we have -1 values (NaN)
    else:
        #for int or float: fill NaN
        tmp_len = len(train[train_series.isnull()])
        if tmp_len>0:
            #print "mean", train_series.mean()
            train.loc[train_series.isnull(), train_name] = -999 
        #and Test
        tmp_len = len(test[test_series.isnull()])
        if tmp_len>0:
            test.loc[test_series.isnull(), test_name] = -999

rf = ensemble.ExtraTreesClassifier(n_jobs=4, 
	n_estimators = n_estimators, 
	random_state = 11, 
	criterion = criterion,
    max_features = max_features,
    min_samples_split= min_leaf_split,
    min_samples_leaf= min_leaf_split,
    max_depth= max_depth)
    
score, predicted = _cv_tools.generic_cv(train, target, rf, n_folds, random_state_cv)

rf.fit(train,target)

np.savetxt("../gen_data/"+model_name+".train.csv", predicted, delimiter=",")

test_pred = rf.predict_proba(test)[:,1]

np.savetxt("../gen_data/"+model_name+".test.csv", test_pred, delimiter=",")
