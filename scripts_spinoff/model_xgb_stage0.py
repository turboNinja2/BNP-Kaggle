import _import_data
import _cv_tools
import numpy as np
import csv
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

train, test, target, test_ids = _import_data.import_stage0()

for n_estimators in [500,1000] :
    for max_depth in [6, 7, 8] :
        for col_sample in [0.8, 1] :
            model = xgb.XGBClassifier(n_estimators=n_estimators,
                nthread=-1,
                max_depth=max_depth,
                learning_rate=0.01,
                silent=True,
                subsample=0.8,
                colsample_bytree=col_sample)
                
            score, _ =_cv_tools.generic_cv(train,target, model, 5 ,1)

            model_name = "xgb_stage0_" + str(score) + "_" + str(n_estimators) + "_" + str(max_depth) + "_" + str(col_sample)
            
            model.fit(train,target)

            test_pred = model.predict_proba(test)[:,1]
            predictions_file = open("../staged_submissions/"+model_name+".csv", "w")
            open_file_object = csv.writer(predictions_file)
            open_file_object.writerow(["ID", "PredictedProb"])
            open_file_object.writerows(zip(test_ids,test_pred))
            predictions_file.close()
        
    '''
model.fit(np.array(train.values),np.array(target.values))
    
test_pred = model.predict_proba(np.array(test.values))[:,1]

predictions_file = open("simple_sgd_blend.csv", "w")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["ID", "PredictedProb"])
open_file_object.writerows(zip(test_ids,test_pred))
predictions_file.close()
'''