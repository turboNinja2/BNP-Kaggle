import _import_data
import _cv_tools
import numpy as np
import xgboost as xgb
import sys

max_depth = 1
arg_parser_index = 0
feature_name = ""

while arg_parser_index < len(sys.argv) :
    if sys.argv[arg_parser_index] == '-feature_name':
        feature_name = sys.argv[arg_parser_index+1]
    arg_parser_index+=1


train, test, target, test_ids = _import_data.import_raw_data()

train_feature = train[feature_name]
test_feature = test[feature_name]

np.savetxt("../gen_data/feat_forward_"+feature_name+".train.csv", train_feature, delimiter=",")
np.savetxt("../gen_data/feat_forward_"+feature_name+".test.csv", test_feature, delimiter=",")
