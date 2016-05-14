import pandas as pd

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

id_test = df_test['ID'].to_csv('../gen_data/test_ids.csv',index=False)
y_train = df_train['target'].to_csv('../gen_data/relevance.csv',index=False)

print('Labels extracted !')
