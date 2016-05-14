import pandas as pd
import numpy as np
from time import time
from sklearn.metrics import log_loss
from sklearn import cross_validation

def generic_cv(X,y,model,n_folds,random_state) :
    kf = cross_validation.KFold(y.shape[0],n_folds=n_folds, shuffle=True, random_state=random_state)
    trscores, cvscores, times = [], [], []
    i = 0
    stack_train = np.zeros((len(y))) # stacked predictions
    for i, (train_fold, validate) in enumerate(kf) :
        i = i + 1
        t = time()
        model.fit(X.iloc[train_fold], y.iloc[train_fold])
        trscore = log_loss(y.iloc[train_fold], model.predict_proba(X.iloc[train_fold])[:,1])
        
        validation_prediction = model.predict_proba(X.iloc[validate])[:,1]
        
        cvscore = log_loss(y.iloc[validate], validation_prediction)
        trscores.append(trscore); cvscores.append(cvscore); times.append(time()-t)
        
        stack_train[validate] = validation_prediction
    
    print("TRAIN %.5f | TEST %.5f | TIME %.2fm (1-fold)" % (np.mean(trscores), np.mean(cvscores), np.mean(times)/60))
    print(model.get_params(deep = True))
    print("\n")
    
    return np.mean(cvscores), stack_train
    

def generic_cv_np(X,y,model,n_folds,random_state) :
    kf = cross_validation.KFold(y.shape[0],n_folds=n_folds, shuffle=True, random_state=random_state)
    trscores, cvscores, times = [], [], []
    i = 0
    stack_train = np.zeros((len(y))) # stacked predictions
    for i, (train_fold, validate) in enumerate(kf) :
        i = i + 1
        t = time()
        
        model.fit(X[train_fold,], y[train_fold])
        
        trscore = log_loss(y[train_fold], model.predict_proba(X[train_fold,]))
        
        validation_prediction = model.predict_proba(X[validate,])
        
        cvscore = log_loss(y[validate], validation_prediction)
        trscores.append(trscore); cvscores.append(cvscore); times.append(time()-t)
        
        stack_train[validate] = validation_prediction
    
    print("TRAIN %.5f | TEST %.5f | TIME %.2fm (1-fold)" % (np.mean(trscores), np.mean(cvscores), np.mean(times)/60))
    print(model.get_params())
    print("\n")
    
    return np.mean(cvscores), stack_train
    
    
def generic_cv_reg(X,y,model,n_folds,random_state) :
    kf = cross_validation.KFold(y.shape[0],n_folds=n_folds, shuffle=True, random_state=random_state)
    trscores, cvscores, times = [], [], []
    i = 0
    stack_train = np.zeros((len(y))) # stacked predictions
    
    threshold = 0.000001
    
    for i, (train_fold, validate) in enumerate(kf) :
        i = i + 1
        t = time()
        trscore = log_loss(y.iloc[train_fold], model.fit(X.iloc[train_fold], y.iloc[train_fold]).predict(X.iloc[train_fold]))
        
        validation_prediction = model.predict(X.iloc[validate])
        
        validation_prediction[validation_prediction>1-threshold] = 1-threshold
        validation_prediction[validation_prediction<threshold] = threshold
        
        cvscore = log_loss(y.iloc[validate], validation_prediction)
        trscores.append(trscore); cvscores.append(cvscore); times.append(time()-t)
        
        stack_train[validate] = validation_prediction
    
    print("TRAIN %.5f | TEST %.5f | TIME %.2fm (1-fold)" % (np.mean(trscores), np.mean(cvscores), np.mean(times)/60))
    print(model.get_params(deep = True))
    print("\n")
    
    return np.mean(cvscores), stack_train