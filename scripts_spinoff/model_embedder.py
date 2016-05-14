import numpy as np

class OneHotNaiveBayes :

    def __init__(self, feature_name, id_name) :
        self.feature_name = feature_name
        self.id_name = id_name

        self.col_with_target = None
        self.targets_sum = None
        
    def fit(self, train, target):
        train = train.copy()
        
        curren_feature_data = train[self.feature_name]
        self.col_with_target = pd.concat([curren_feature_data, target],axis=1,ignore_index=True)
    
        ctrain, ctest = frozenset(train[self.feature_name]), frozenset(test[self.feature_name])

        cboth = ctrain.intersection(ctest)

        train.loc[~train[self.feature_name].isin(cboth), self.feature_name] = np.nan
        test.loc[~test[self.feature_name].isin(cboth), self.feature_name] = np.nan

        target_values = col_with_target[target_column].unique()
        for t in target_values:
            col_with_target[c + '_targets_' + str(t)] = 1.0 * (col_with_target[target_column] == t)
        col_with_target.drop([target_column], inplace=True, axis=1)
        
        self.targets_sum = col_with_target.groupby(c).sum().apply(lambda r: r / r.sum(), axis=1, raw=True)

        
        
        
    def predict_proba(self,test) :
        test = test.copy()
        
        test = pd.merge(test, self.targets_sum, left_on=self.feature_name, right_index=True, how='left')
    
        test_preds = test[self.feature_name + '_targets_1'].fillna(test[self.feature_name + '_targets_1'].mean())
    
        return test_preds
        
    def get_params(self, deep = True):
        return self.model.get_params(deep)