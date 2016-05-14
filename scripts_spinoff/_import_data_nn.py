import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from keras.utils import np_utils, generic_utils

def scale_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()       
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

def getDummiesInplace(columnList, train, test):
    #Takes in a list of column names and one or two pandas dataframes
    #One-hot encodes all indicated columns inplace
    columns = []
    
    df = pd.concat([train,test], axis= 0)

    for column_name in df.columns:
        index = df.columns.get_loc(column_name)
        if column_name in columnList:
            dummies = pd.get_dummies(df.ix[:,index], prefix = column_name, prefix_sep = ".")
            columns.append(dummies)
        else:
            columns.append(df.ix[:,index])
    df = pd.concat(columns, axis = 1)
    
    train = df[:train.shape[0]]
    test = df[train.shape[0]:]
    return train, test

        
def pdFillNAN(df, strategy = "mean"):
    if strategy == "mean":
        return df.fillna(df.mean())
    elif type(strategy) == int:
        return df.fillna(strategy)

def gen_data_1(seed=1) :

    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')


    #Drop target, ID, and v22(due to too many levels)
    target = train["target"]
    test_ids = test["ID"]
    train.drop(labels = "ID", axis = 1, inplace = True)
    train.drop(labels = "target", axis = 1, inplace = True)
    test.drop(labels = "ID", axis = 1, inplace = True)
    train.drop(labels = "v22", axis = 1, inplace = True)
    test.drop(labels = "v22", axis = 1, inplace = True)

    #find categorical variables
    categoricalVariables = []
    for var in train.columns:
        vector=pd.concat([train[var],test[var]], axis=0)
        typ=str(train[var].dtype)
        if (typ=='object'):
            categoricalVariables.append(var)
            
    train, test = getDummiesInplace(categoricalVariables, train, test)

    #Remove sparse columns
    cls = train.sum(axis=0)
    train = train.drop(train.columns[cls<10], axis=1)
    test = test.drop(test.columns[cls<10], axis=1)

    fillNANStrategy = -1
    #fillNANStrategy = "mean"
    train = pdFillNAN(train, fillNANStrategy)
    test = pdFillNAN(test, fillNANStrategy)


    train, scaler = scale_data(train)
    test, scaler = scale_data(test, scaler)

    encoder = LabelEncoder()    
    target = encoder.fit_transform(target).astype(np.int32)
    target = np_utils.to_categorical(target)
            
    return train, test, target, test_ids        
            
            
            