import _import_data
import _cv_tools

import numpy as np
import csv
import sys

from time import time

from sklearn.metrics import log_loss
from sklearn import cross_validation
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss, auc, roc_auc_score
from keras.optimizers import Adagrad,SGD,Adadelta
from keras.callbacks import EarlyStopping
from keras.layers import containers
from keras.layers.core import Dense, AutoEncoder
from keras.constraints import maxnorm

from keras.utils import np_utils, generic_utils

n_folds = 5
random_state_cv = 1
n_epochs = 10 
first_layer = 10
second_layer = 10
arg_parser_index = 0
third_layer = 10

while arg_parser_index < len(sys.argv) :
    if sys.argv[arg_parser_index] == '-first_layer':
        first_layer = int(sys.argv[arg_parser_index+1])
    if sys.argv[arg_parser_index] == '-second_layer':
        second_layer = int(sys.argv[arg_parser_index+1])
    if sys.argv[arg_parser_index] == '-third_layer':
        third_layer = int(sys.argv[arg_parser_index+1])
    if sys.argv[arg_parser_index] == '-n_epochs':
        n_epochs = int(sys.argv[arg_parser_index+1])
    if sys.argv[arg_parser_index] == '-random_state_cv':
        random_state_cv = int(sys.argv[arg_parser_index+1])
    arg_parser_index+=1

train, test, target, test_ids = _import_data.import_stage0()

train = train.fillna(target.mean())
test = test.fillna(target.mean())

train = train.as_matrix()
test = test.as_matrix()

encoder = LabelEncoder()    
target = encoder.fit_transform(target).astype(np.int32)
target = np_utils.to_categorical(target)

dims = train.shape[1]
nb_classes = target.shape[1]

kf = cross_validation.KFold(target.shape[0],n_folds=n_folds, shuffle=True, random_state=random_state_cv)
trscores, cvscores, times = [], [], []
i = 0
stack_train = np.zeros((len(target))) # stacked predictions

for i, (train_fold, validate) in enumerate(kf) :
    i = i + 1
    t = time()
        
    model = Sequential()

    model.add(Dense(first_layer, input_shape=(dims,)))
    model.add(Dropout(0.1))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(second_layer))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(third_layer))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy', optimizer="sgd")   
        
    model.fit(train[train_fold,], target[train_fold],batch_size=128,nb_epoch=n_epochs, verbose = 0)
        
    trscore = log_loss(target[train_fold], model.predict_proba(train[train_fold,], verbose = 0)[:,1])
        
    validation_prediction = model.predict_proba(train[validate,], verbose = 0)[:,1]
    
    
    cvscore = log_loss(target[validate], validation_prediction)
    trscores.append(trscore); cvscores.append(cvscore); times.append(time()-t)
        
    stack_train[validate] = validation_prediction
        
print("TRAIN %.5f | TEST %.5f | TIME %.2fm (1-fold)" % (np.mean(trscores), np.mean(cvscores), np.mean(times)/60))
print("\n")