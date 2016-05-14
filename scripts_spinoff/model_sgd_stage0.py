import _import_data
import _cv_tools
import numpy as np
import csv
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV

train, test, target, test_ids = _import_data.import_stage0()

model = SGDClassifier(loss="log",
                        penalty="elasticnet", 
                        n_jobs = -1,
                        n_iter = 100,
                        random_state = 123)
        
model = CalibratedClassifierCV(base_estimator=model, cv=5, method='isotonic')

_cv_tools.generic_cv(train,target, model, 5 ,1)

model.fit(np.array(train.values),np.array(target.values))
    
test_pred = model.predict_proba(np.array(test.values))[:,1]

predictions_file = open("simple_sgd_blend.csv", "w")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["ID", "PredictedProb"])
open_file_object.writerows(zip(test_ids,test_pred))
predictions_file.close()