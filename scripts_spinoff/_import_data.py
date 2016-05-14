import next_prime
import glob
import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics

    
def interactions(dataframe,var_name_1,var_name_2):
    dataframe[var_name_1 + '_' + var_name_2] = dataframe[var_name_1].map(str) + dataframe[var_name_2].map(str)
    return dataframe

    
def shift_factors(train,test,offset):
    for f in train.columns:
        if train[f].dtype=='object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[f].values) + list(test[f].values))
            train[f] = lbl.transform(list(train[f].values))
            test[f] = lbl.transform(list(test[f].values))
            
            largest_label_val = max(max(train[f]),max(test[f]))
            modulo = next_prime.next_prime(largest_label_val + offset)
            train[f] = train[f]*offset % modulo
            test[f] = test[f]*offset % modulo
    return train,test  

def replace_factors_by_response(train,test):
    mean_response = train.target.mean()
    rare_count = 20
    
    for feat in train.columns:
        if feat != 'target' :
            if train[feat].dtype=='object' or test[feat].dtype=='object' :
            
                train.loc[train[feat].value_counts()[train[feat]].values <  rare_count, feat] = "RARE"
                test.loc[train[feat].value_counts()[train[feat]].values  <  rare_count, feat] = "RARE"
                
                criterion = ~test[feat].isin(set(train[feat]))
                
                test.loc[criterion,feat] = "RARE"
                
                m = train.groupby([feat])['target'].mean()
                
                train[feat] = train[feat].replace(m, inplace=False)
                test[feat]  = test[feat].replace(m, inplace=False)
               
    return train, test
    
def select_important_features_1(train,test):
    important_features = ['v50','v10','v12','v22','v21','v34','v14','v40','v114','v66','v125','v112','v52','v47','v56',
        'v113','v31','v24','v91','v107','v30','v79','v82','v23','v62','v98','v120','v28','v6']
    
    train = train[important_features]    
    test = test[important_features]    
        
    return train, test
    
def import_train_test() :
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    return train, test
    
def import_raw_data():
    train, test = import_train_test()
    
    target = train['target']
    train = train.drop(['ID','target'],axis=1)
    
    test_ids = test['ID'].values
    test = test.drop(['ID'],axis=1)
    
    return train, test, target, test_ids

def factorize(train,test) :
    print('Clearing...')
    
    for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
        if train_series.dtype == 'O':
            train[train_name], tmp_indexer = pd.factorize(train[train_name])
            test[test_name] = tmp_indexer.get_indexer(test[test_name])
        else:
            tmp_len = len(train[train_series.isnull()])
            if tmp_len>0:
                train.loc[train_series.isnull(), train_name] = train_series.mean()
            tmp_len = len(test[test_series.isnull()])
            if tmp_len>0:
                test.loc[test_series.isnull(), test_name] = train_series.mean()  #TODO
    return train, test    
    
  
    
def gen_data_1():
    train, test, target, test_ids = import_raw_data()   
    train,test = factorize(train,test)
    return train, test, target, test_ids

    
def gen_data_1_sub():
    train, test, target, test_ids = import_raw_data()   
    train, test = select_important_features_1(train,test)
    train,test = factorize(train,test)
    return train, test, target, test_ids
    
    
def gen_data_2(offset):
    train, test, target, test_ids = import_raw_data()   
    print('Running offset : ' + str(offset))
    train, test = shift_factors(train,test,offset)
    return train, test, target, test_ids


def gen_data_3(offset):
    train, test, target, test_ids = import_raw_data()   
    
    train = interactions(train,'v110','v40')
    test = interactions(test,'v110','v40')

    train = interactions(train,'v47','v40')
    test = interactions(test,'v47','v40')
    
    train = interactions(train,'v47','v12')
    test = interactions(test,'v47','v12')

    train = interactions(train,'v79','v40')
    test = interactions(test,'v79','v40')
    
    print('Running offset : ' + str(offset))
    train, test = shift_factors(train,test,offset)
    return train, test, target, test_ids

    
def expand_factors(train,test,max_categories,target) :
    traindummies=pd.DataFrame()
    testdummies=pd.DataFrame()
    
    for elt in train.columns:
        vector=pd.concat([train[elt],test[elt]], axis=0)

        #count as categorial if number of unique values is less than max_categories
        if len(vector.unique())<max_categories:
            traindummies=pd.concat([traindummies, pd.get_dummies(train[elt],prefix=elt,dummy_na=True)], axis=1).astype('int8')
            testdummies=pd.concat([testdummies, pd.get_dummies(test[elt],prefix=elt,dummy_na=True)], axis=1).astype('int8')
            del train[elt], test[elt]
        else:
            typ=str(train[elt].dtype)[:3]
            if (typ=='flo') or (typ=='int'):
                minimum=vector.min()
                maximum=vector.max()
                train[elt]=train[elt].fillna(int(minimum)-2)
                test[elt]=test[elt].fillna(int(minimum)-2)
                minimum=int(minimum)-2
                traindummies[elt+'_na']=train[elt].apply(lambda x: 1 if x==minimum else 0)
                testdummies[elt+'_na']=test[elt].apply(lambda x: 1 if x==minimum else 0)
                

                #resize between 0 and 1 linearly ax+b
                a=1/(maximum-minimum)
                b=-a*minimum
                train[elt]=a*train[elt]+b
                test[elt]=a*test[elt]+b
            else:
                if (typ=='obj'):
                    list2keep=vector.value_counts()[:max_categories].index
                    train[elt]=train[elt].apply(lambda x: x if x in list2keep else np.nan)
                    test[elt]=test[elt].apply(lambda x: x if x in list2keep else np.nan)                
                    traindummies=pd.concat([traindummies, pd.get_dummies(train[elt],prefix=elt,dummy_na=True)], axis=1).astype('int8')
                    testdummies=pd.concat([testdummies, pd.get_dummies(test[elt],prefix=elt,dummy_na=True)], axis=1).astype('int8')
                    
                    #Replace categories by their weights
                    tempTable=pd.concat([train[elt], target], axis=1)
                    tempTable=tempTable.groupby(by=elt, axis=0).agg(['sum','count']).target
                    tempTable['weight']=tempTable.apply(lambda x: .5+.5*x['sum']/x['count'] if (x['sum']>x['count']-x['sum']) else .5+.5*(x['sum']-x['count'])/x['count'], axis=1)
                    tempTable.reset_index(inplace=True)
                    train[elt+'weight']=pd.merge(train, tempTable, how='left', on=elt)['weight']
                    test[elt+'weight']=pd.merge(test, tempTable, how='left', on=elt)['weight']
                    train[elt+'weight']=train[elt+'weight'].fillna(.5)
                    test[elt+'weight']=test[elt+'weight'].fillna(.5)
                    del train[elt], test[elt]
                else:
                    print('error', typ)

    #remove na values too similar to v2_na
    
    for elt in train.columns:
        if (elt[-2:]=='na') & (elt!='v2_na'):
            dist=metrics.pairwise_distances(train.v2_na.reshape(1, -1),train[elt].reshape(1, -1))
            if dist<8:
                del train[elt],test[elt]
            else:
                print(elt, dist)
                
                
    train=pd.concat([train,traindummies, target], axis=1)
    test=pd.concat([test,testdummies], axis=1)
    
    #remove features only present in train or test
    for elt in list(set(train.columns)-set(test.columns)):
        del train[elt]
    for elt in list(set(test.columns)-set(train.columns)):
        del test[elt]
    
    return train,test

def gen_data_4(max_categories):
    train, test, target, test_ids = import_raw_data()   
    train, test = expand_factors(train,test,max_categories,target)
    return train, test, target, test_ids
    

def gen_data_5(max_categories):
    train, test, target, test_ids = import_raw_data()   
    
    train = interactions(train,'v110','v40')
    test = interactions(test,'v110','v40')

    train = interactions(train,'v47','v40')
    test = interactions(test,'v47','v40')
    
    train = interactions(train,'v47','v12')
    test = interactions(test,'v47','v12')

    train = interactions(train,'v79','v40')
    test = interactions(test,'v79','v40')
    
    train, test = expand_factors(train,test,max_categories,target)
    
    return train, test, target, test_ids
    

def gen_data_6(max_categories):
    train, test, target, test_ids = import_raw_data()   
    
    train = interactions(train,'v110','v40')
    test = interactions(test,'v110','v40')

    train = interactions(train,'v47','v40')
    test = interactions(test,'v47','v40')
    
    train = interactions(train,'v47','v12')
    test = interactions(test,'v47','v12')

    train = interactions(train,'v79','v40')
    test = interactions(test,'v79','v40')
    
    train = interactions(train,'v110','v47')
    test = interactions(test,'v110','v47')
    
    train = interactions(train,'v22','v40')
    test = interactions(test,'v22','v40')
    
    train, test = expand_factors(train,test,max_categories,target)
    
    return train, test, target, test_ids


def gen_data_7() :
    print('Load data...')
    train, test, target, test_ids = import_raw_data()

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
                train.loc[train_series.isnull(), train_name] = -9999 #train_series.mean()
            #and Test
            tmp_len = len(test[test_series.isnull()])
            if tmp_len>0:
                test.loc[test_series.isnull(), test_name] = -9999 #train_series.mean()  #TODO

    return train, test, target, test_ids

def gen_data_8() :
    train, test = import_train_test()

    
    train = train.drop(['ID'],axis=1)
    
    test_ids = test['ID'].values
    test = test.drop(['ID'],axis=1)
    
    train = train.fillna(-1)
    test = test.fillna(-1)
    
    train,test = replace_factors_by_response(train, test)
    
    target = train['target']
    train = train.drop(['target'],axis=1)
    
    
    return train, test, target, test_ids
 
def import_features(path = '../gen_data/*.train.csv'):
    train_files = glob.glob(path)
    train_file = train_files.pop(0)
    train_set = pd.read_csv(train_file,header=None)
    
    for train_file in train_files:
        current_data = pd.read_csv(train_file,header=None)
        train_set = pd.concat([train_set, current_data],axis=1,ignore_index=True)
    
    return train_set
    
    
def import_stage0(path = '../gen_data/'):
    _, _, target, test_ids = import_raw_data()   
    
    train = import_features(path + '*.train.csv')
    test  = import_features(path + '*.test.csv')
    
    return train, test, target, test_ids
