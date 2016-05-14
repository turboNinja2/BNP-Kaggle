del *.pyc
del .Rhistory

python extract_labels.py

python model_1.py -max_depth 5 -n_estimators 200 -random_state_cv 211
python model_1.py -max_depth 6 -n_estimators 200 -random_state_cv 212
python model_1.py -max_depth 7 -n_estimators 200 -random_state_cv 213
python model_1.py -max_depth 8 -n_estimators 200 -random_state_cv 214

python model_1.py -max_depth 9 -n_estimators 300 -random_state_cv 215
python model_1.py -max_depth 10 -n_estimators 300 -random_state_cv 216
python model_1.py -max_depth 15 -n_estimators 300 -random_state_cv 217

python model_xgb_2.py -max_depth 6 -n_estimators 300 -offset 1 -random_state_cv 218
python model_xgb_2.py -max_depth 8 -n_estimators 300 -offset 3 -random_state_cv 219
python model_xgb_2.py -max_depth 10 -n_estimators 300 -offset 5 -random_state_cv 2110
python model_xgb_2.py -max_depth 12 -n_estimators 300 -offset 11 -random_state_cv 2111

python model_rf_1.py -criterion entropy -n_estimators 200 -random_state_cv 2112
python model_rf_1.py -criterion entropy -n_estimators 500 -random_state_cv 2113
python model_rf_1.py -criterion gini -n_estimators 200 -random_state_cv 2114
python model_rf_1.py -criterion gini -n_estimators 500 -random_state_cv 2115

python model_xgb_2.py -max_depth 6 -n_estimators 300 -offset 7 -random_state_cv 2116
python model_xgb_2.py -max_depth 8 -n_estimators 300 -offset 23 -random_state_cv 2117
python model_xgb_2.py -max_depth 10 -n_estimators 300 -offset 29 -random_state_cv 2118

python model_xgb_2.py -max_depth 6 -n_estimators 300 -offset 31 -random_state_cv 2119
python model_xgb_2.py -max_depth 8 -n_estimators 300 -offset 37 -random_state_cv 2120
python model_xgb_2.py -max_depth 10 -n_estimators 300 -offset 39 -random_state_cv 2121


python model_xgb_2.py -max_depth 6 -n_estimators 300 -offset 7 -random_state_cv 2122
python model_xgb_2.py -max_depth 8 -n_estimators 300 -offset 23 -random_state_cv 2123
python model_xgb_2.py -max_depth 10 -n_estimators 300 -offset 29 -random_state_cv 2124

python model_xgb_2.py -max_depth 6 -n_estimators 300 -offset 31 -random_state_cv 2125
python model_xgb_2.py -max_depth 8 -n_estimators 300 -offset 37 -random_state_cv 2126
python model_xgb_2.py -max_depth 10 -n_estimators 300 -offset 39 -random_state_cv 2127

python model_xgb_4.py -n_estimators 300 -max_depth 8 -random_state_cv 2134
python model_xgb_4.py -n_estimators 500 -max_depth 8 -random_state_cv 2135

python model_xgb_5.py -max_depth 7 -n_estimators 2000 -offset 2 -random_state_cv 2136
python model_xgb_5.py -max_depth 8 -n_estimators 2000 -offset 13 -random_state_cv 2137
python model_xgb_5.py -max_depth 9 -n_estimators 2000 -offset 17 -random_state_cv 2138

python model_erf_1.py -criterion entropy -n_estimators 200 -random_state_cv 2139
python model_erf_1.py -criterion entropy -n_estimators 500 -random_state_cv 2140
python model_erf_1.py -criterion gini -n_estimators 200 -random_state_cv 2141
python model_erf_1.py -criterion gini -n_estimators 500 

python model_erf_2.py -criterion entropy -n_estimators 200 -max_categories 10 -random_state_cv 2142
python model_erf_2.py -criterion entropy -n_estimators 500 -max_categories 20 -random_state_cv 2143
python model_erf_2.py -criterion gini -n_estimators 200 -max_categories 15 -random_state_cv 2144
python model_erf_2.py -criterion gini -n_estimators 500 -max_categories 25 -random_state_cv 2145

python model_rf_2.py -criterion entropy -n_estimators 200 -max_categories 10 -random_state_cv 2146
python model_rf_2.py -criterion entropy -n_estimators 500 -max_categories 25 -random_state_cv 2147
python model_rf_2.py -criterion gini -n_estimators 200 -max_categories 15 -random_state_cv 2148
python model_rf_2.py -criterion gini -n_estimators 500 -max_categories 25 -random_state_cv 2149

python model_xgb_6.py -max_depth 7 -n_estimators 300 -offset 7 -random_state_cv 2150
python model_xgb_6.py -max_depth 8 -n_estimators 400 -offset 23 -random_state_cv 2151
python model_xgb_6.py -max_depth 9 -n_estimators 500 -offset 29 -random_state_cv 2152

python model_xgb_1_sub.py -max_depth 6 -n_estimators 200 -random_state_cv 2153
python model_xgb_1_sub.py -max_depth 8 -n_estimators 200 -random_state_cv 2154
python model_xgb_1_sub.py -max_depth 20 -n_estimators 200 -random_state_cv 2155


python model_erf_3.py -criterion entropy -n_estimators 200 -max_categories 13 -random_state_cv 2156
python model_erf_3.py -criterion entropy -n_estimators 500 -max_categories 23 -random_state_cv 2157
python model_erf_3.py -criterion gini -n_estimators 200 -max_categories 18 -random_state_cv 2158
python model_erf_3.py -criterion gini -n_estimators 500 -max_categories 28 -random_state_cv 2159

python model_rf_3.py -criterion entropy -n_estimators 200 -random_state_cv 2160 
python model_rf_3.py -criterion entropy -n_estimators 500 -random_state_cv 2161
python model_rf_3.py -criterion gini -n_estimators 200 -random_state_cv 2162
python model_rf_3.py -criterion gini -n_estimators 500 -random_state_cv 2163

python model_sgd_1.py -max_categories 20 -random_state_cv 2164
python model_sgd_1.py -max_categories 60 -random_state_cv 2165
python model_sgd_2.py -max_categories 80 -random_state_cv 2166

python model_ohnb.py -var_name v22 -random_state_cv 2167
python model_ohnb.py -var_name v72 -random_state_cv 2168
python model_ohnb.py -var_name v38 -random_state_cv 2169
python model_ohnb.py -var_name v113 -random_state_cv 2170


python model_nn_1.py -first_layer 200 -second_layer 10 -n_epochs 100 -random_state_cv 2181
python model_nn_1.py -first_layer 400 -second_layer 50 -n_epochs 100 -random_state_cv 2182
python model_nn_1.py -first_layer 100 -second_layer 200 -n_epochs 100 -random_state_cv 2183
python model_nn_1.py -first_layer 400 -second_layer 100 -n_epochs 100 -random_state_cv 2184

python model_nn_2.py -first_layer 128 -second_layer 64 -third_layer 32 -n_epochs 120 -random_state_cv 2185
python model_nn_2.py -first_layer 64 -second_layer 64 -third_layer 64 -n_epochs 150 -random_state_cv 2186

python model_ohnb_interaction.py -var_name1 v66 -var_name2 v113 -random_state_cv 2187
python model_serf_1.py -max_depth 20 -max_features 50 -n_estimators 800 -random_state_cv 2188

python model_nn_2.py -first_layer 64 -second_layer 32 -third_layer 64 -n_epochs 150 -random_state_cv 2189
python model_nn_2.py -first_layer 128 -second_layer 64 -third_layer 64 -n_epochs 150 -random_state_cv 2190
python model_nn_2.py -first_layer 256 -second_layer 128 -third_layer 64 -n_epochs 150 -random_state_cv 2191

python feature_forwarding.py -feature_name v50
python feature_forwarding.py -feature_name v10
python feature_forwarding.py -feature_name v12
python feature_forwarding.py -feature_name v21
python feature_forwarding.py -feature_name v114


python model_dt_1.py -criterion entropy -max_depth 5 -random_state_cv 2192
python model_dt_1.py -criterion entropy -max_depth 3 -random_state_cv 2193

python model_dt_1.py -criterion gini -max_depth 5 -random_state_cv 2194
python model_dt_1.py -criterion gini -max_depth 3 -random_state_cv 2195


del *.pyc
del .Rhistory

pause