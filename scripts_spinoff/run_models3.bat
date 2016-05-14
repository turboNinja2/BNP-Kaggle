del *.pyc
del .Rhistory

echo python extract_labels.py

echo python model_1.py -max_depth 5 -n_estimators 200 -random_state_cv 2116
echo python model_1.py -max_depth 6 -n_estimators 200 -random_state_cv 2126
echo python model_1.py -max_depth 7 -n_estimators 200 -random_state_cv 2136
python model_1.py -max_depth 8 -n_estimators 200 -random_state_cv 2146

python model_1.py -max_depth 9 -n_estimators 300 -random_state_cv 2156
python model_1.py -max_depth 10 -n_estimators 300 -random_state_cv 2166
python model_1.py -max_depth 15 -n_estimators 300 -random_state_cv 2176

python model_xgb_2.py -max_depth 6 -n_estimators 300 -offset 1 -random_state_cv 2186
python model_xgb_2.py -max_depth 8 -n_estimators 300 -offset 3 -random_state_cv 2196
python model_xgb_2.py -max_depth 10 -n_estimators 300 -offset 5 -random_state_cv 21106
python model_xgb_2.py -max_depth 12 -n_estimators 300 -offset 11 -random_state_cv 21116

python model_rf_1.py -criterion entropy -n_estimators 200 -random_state_cv 21126
python model_rf_1.py -criterion entropy -n_estimators 500 -random_state_cv 21136
python model_rf_1.py -criterion gini -n_estimators 200 -random_state_cv 21146
python model_rf_1.py -criterion gini -n_estimators 500 -random_state_cv 21156

python model_xgb_2.py -max_depth 6 -n_estimators 300 -offset 7 -random_state_cv 21166
python model_xgb_2.py -max_depth 8 -n_estimators 300 -offset 23 -random_state_cv 21176
python model_xgb_2.py -max_depth 10 -n_estimators 300 -offset 29 -random_state_cv 21186

python model_xgb_2.py -max_depth 6 -n_estimators 300 -offset 31 -random_state_cv 21196
python model_xgb_2.py -max_depth 8 -n_estimators 300 -offset 37 -random_state_cv 21206
python model_xgb_2.py -max_depth 10 -n_estimators 300 -offset 39 -random_state_cv 21216


python model_xgb_2.py -max_depth 6 -n_estimators 300 -offset 7 -random_state_cv 21226
python model_xgb_2.py -max_depth 8 -n_estimators 300 -offset 23 -random_state_cv 21236
python model_xgb_2.py -max_depth 10 -n_estimators 300 -offset 29 -random_state_cv 21246

python model_xgb_2.py -max_depth 6 -n_estimators 300 -offset 31 -random_state_cv 21256
python model_xgb_2.py -max_depth 8 -n_estimators 300 -offset 37 -random_state_cv 21266
python model_xgb_2.py -max_depth 10 -n_estimators 300 -offset 39 -random_state_cv 21276

python model_xgb_4.py -n_estimators 300 -max_depth 8 -random_state_cv 21346
python model_xgb_4.py -n_estimators 500 -max_depth 8 -random_state_cv 21356

python model_xgb_5.py -max_depth 7 -n_estimators 2000 -offset 2 -random_state_cv 21366
python model_xgb_5.py -max_depth 8 -n_estimators 2000 -offset 13 -random_state_cv 21376
python model_xgb_5.py -max_depth 9 -n_estimators 2000 -offset 17 -random_state_cv 21386

python model_erf_1.py -criterion entropy -n_estimators 200 -random_state_cv 21396
python model_erf_1.py -criterion entropy -n_estimators 500 -random_state_cv 21406
python model_erf_1.py -criterion gini -n_estimators 200 -random_state_cv 21416
python model_erf_1.py -criterion gini -n_estimators 500 -random_state_cv 66

python model_erf_2.py -criterion entropy -n_estimators 200 -max_categories 10 -random_state_cv 21426
python model_erf_2.py -criterion entropy -n_estimators 500 -max_categories 20 -random_state_cv 21436
python model_erf_2.py -criterion gini -n_estimators 200 -max_categories 15 -random_state_cv 21446
python model_erf_2.py -criterion gini -n_estimators 500 -max_categories 25 -random_state_cv 21456

python model_rf_2.py -criterion entropy -n_estimators 200 -max_categories 10 -random_state_cv 21466
python model_rf_2.py -criterion entropy -n_estimators 500 -max_categories 25 -random_state_cv 21476
python model_rf_2.py -criterion gini -n_estimators 200 -max_categories 15 -random_state_cv 21486
python model_rf_2.py -criterion gini -n_estimators 500 -max_categories 25 -random_state_cv 21496

python model_xgb_6.py -max_depth 7 -n_estimators 300 -offset 7 -random_state_cv 21506
python model_xgb_6.py -max_depth 8 -n_estimators 400 -offset 23 -random_state_cv 21516
python model_xgb_6.py -max_depth 9 -n_estimators 500 -offset 29 -random_state_cv 21526

python model_xgb_1_sub.py -max_depth 6 -n_estimators 200 -random_state_cv 21536
python model_xgb_1_sub.py -max_depth 8 -n_estimators 200 -random_state_cv 21546
python model_xgb_1_sub.py -max_depth 20 -n_estimators 200 -random_state_cv 21556


python model_erf_3.py -criterion entropy -n_estimators 200 -max_categories 13 -random_state_cv 21566
python model_erf_3.py -criterion entropy -n_estimators 500 -max_categories 23 -random_state_cv 21576
python model_erf_3.py -criterion gini -n_estimators 200 -max_categories 18 -random_state_cv 21586
python model_erf_3.py -criterion gini -n_estimators 500 -max_categories 28 -random_state_cv 21596

python model_rf_3.py -criterion entropy -n_estimators 200 -random_state_cv 21606
python model_rf_3.py -criterion entropy -n_estimators 500 -random_state_cv 21616
python model_rf_3.py -criterion gini -n_estimators 200 -random_state_cv 21626
python model_rf_3.py -criterion gini -n_estimators 500 -random_state_cv 21636

python model_sgd_1.py -max_categories 20 -random_state_cv 21646
python model_sgd_1.py -max_categories 60 -random_state_cv 21656
python model_sgd_2.py -max_categories 80 -random_state_cv 21666

python model_ohnb.py -var_name v22 -random_state_cv 21676
python model_ohnb.py -var_name v72 -random_state_cv 21686
python model_ohnb.py -var_name v38 -random_state_cv 21696
python model_ohnb.py -var_name v113 -random_state_cv 21706


python model_nn_1.py -first_layer 200 -second_layer 10 -n_epochs 100 -random_state_cv 21816
python model_nn_1.py -first_layer 400 -second_layer 50 -n_epochs 100 -random_state_cv 21826
python model_nn_1.py -first_layer 100 -second_layer 200 -n_epochs 100 -random_state_cv 21836
python model_nn_1.py -first_layer 400 -second_layer 100 -n_epochs 100 -random_state_cv 21846

python model_nn_2.py -first_layer 128 -second_layer 64 -third_layer 32 -n_epochs 120 -random_state_cv 21856
python model_nn_2.py -first_layer 64 -second_layer 64 -third_layer 64 -n_epochs 150 -random_state_cv 21866

python model_ohnb_interaction.py -var_name1 v66 -var_name2 v113 -random_state_cv 21876
python model_serf_1.py -max_depth 20 -max_features 50 -n_estimators 800 -random_state_cv 21886

python model_nn_2.py -first_layer 64 -second_layer 32 -third_layer 64 -n_epochs 150 -random_state_cv 21896
python model_nn_2.py -first_layer 128 -second_layer 64 -third_layer 64 -n_epochs 150 -random_state_cv 21906
python model_nn_2.py -first_layer 256 -second_layer 128 -third_layer 64 -n_epochs 150 -random_state_cv 21916

python feature_forwarding.py -feature_name v50
python feature_forwarding.py -feature_name v10
python feature_forwarding.py -feature_name v12
python feature_forwarding.py -feature_name v21
python feature_forwarding.py -feature_name v114


python model_dt_1.py -criterion entropy -max_depth 5 -random_state_cv 21926
python model_dt_1.py -criterion entropy -max_depth 3 -random_state_cv 21936

python model_dt_1.py -criterion gini -max_depth 5 -random_state_cv 21946
python model_dt_1.py -criterion gini -max_depth 3 -random_state_cv 21956


del *.pyc
del .Rhistory

pause