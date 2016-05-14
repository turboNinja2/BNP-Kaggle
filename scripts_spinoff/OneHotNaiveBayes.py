import numpy as np
import pandas as pd

class OneHotNaiveBayes :

    def __init__(self, feature_name, target_name) :
        self.feature_name = feature_name
        self.target_name = target_name
        
    def fit(self, train, target):
        self.train = train.copy()
        self.target = target.copy()
        feat = self.feature_name
        return self
        
    def predict(self,test) :
        train = self.train
        target = self.target
        test = test.copy()
    
        curren_feature_data = train[self.feature_name]
        col_with_target = pd.concat([curren_feature_data, target],axis=1,ignore_index=True)
    
        col_with_target.columns = [self.feature_name, self.target_name]
    
        ctrain, ctest = frozenset(train[self.feature_name]), frozenset(test[self.feature_name])

        cboth = ctrain.intersection(ctest)

        train.loc[~train[self.feature_name].isin(cboth), self.feature_name] = np.nan
        test.loc[~test[self.feature_name].isin(cboth), self.feature_name] = np.nan

        target_values = target.unique()
        
        for t in target_values:
            col_with_target[self.feature_name + '_targets_' + str(t)] = 1.0 * (target == t)
        
        col_with_target.drop([self.target_name], inplace=True, axis=1)
        
        self.targets_sum = col_with_target.groupby(self.feature_name).sum().apply(lambda r: r / r.sum(), axis=1, raw=True)
    
        test = test.copy()
        
        test = pd.merge(test, self.targets_sum, left_on=self.feature_name, right_index=True, how='left')
    
        test_preds = test[self.feature_name + '_targets_1'].fillna(test[self.feature_name + '_targets_1'].mean())

    
        return test_preds.as_matrix()
        
    def get_params(self, deep = True):
        return self.feature_name