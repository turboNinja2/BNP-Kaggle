del *.pyc
del .Rhistory

python extract_labels.py

python model_1.py -max_depth 5 -n_estimators 200 -random_state_cv 21216
python model_1.py -max_depth 6 -n_estimators 200 -random_state_cv 21226
python model_1.py -max_depth 7 -n_estimators 200 -random_state_cv 21336
python model_1.py -max_depth 8 -n_estimators 200 -random_state_cv 21436

python model_1.py -max_depth 9 -n_estimators 300 -random_state_cv 21356
python model_1.py -max_depth 10 -n_estimators 300 -random_state_cv 21366
python model_1.py -max_depth 15 -n_estimators 300 -random_state_cv 21736

python model_xgb_2.py -max_depth 6 -n_estimators 300 -offset 1 -random_state_cv 21386
python model_xgb_2.py -max_depth 8 -n_estimators 300 -offset 3 -random_state_cv 21936
python model_xgb_2.py -max_depth 10 -n_estimators 300 -offset 5 -random_state_cv 211306
python model_xgb_2.py -max_depth 12 -n_estimators 300 -offset 11 -random_state_cv 211316

python model_rf_1.py -criterion entropy -n_estimators 200 -random_state_cv 2113326
python model_rf_1.py -criterion entropy -n_estimators 500 -random_state_cv 211336
python model_rf_1.py -criterion gini -n_estimators 200 -random_state_cv 211436
python model_rf_1.py -criterion gini -n_estimators 500 -random_state_cv 211356

python model_xgb_2.py -max_depth 6 -n_estimators 300 -offset 7 -random_state_cv 211366
python model_xgb_2.py -max_depth 8 -n_estimators 300 -offset 23 -random_state_cv 211736
python model_xgb_2.py -max_depth 10 -n_estimators 300 -offset 29 -random_state_cv 231186

python model_xgb_2.py -max_depth 6 -n_estimators 300 -offset 31 -random_state_cv 211396
python model_xgb_2.py -max_depth 8 -n_estimators 300 -offset 37 -random_state_cv 231206
python model_xgb_2.py -max_depth 10 -n_estimators 300 -offset 39 -random_state_cv 213216


python model_xgb_2.py -max_depth 6 -n_estimators 300 -offset 7 -random_state_cv 212326
python model_xgb_2.py -max_depth 8 -n_estimators 300 -offset 23 -random_state_cv 213236
python model_xgb_2.py -max_depth 10 -n_estimators 300 -offset 29 -random_state_cv 213246

python model_xgb_2.py -max_depth 6 -n_estimators 300 -offset 31 -random_state_cv 213256
python model_xgb_2.py -max_depth 8 -n_estimators 300 -offset 37 -random_state_cv 212636
python model_xgb_2.py -max_depth 10 -n_estimators 300 -offset 39 -random_state_cv 212376

python model_xgb_4.py -n_estimators 300 -max_depth 8 -random_state_cv 213346
python model_xgb_4.py -n_estimators 500 -max_depth 8 -random_state_cv 213536

python model_xgb_5.py -max_depth 7 -n_estimators 2000 -offset 2 -random_state_cv 213366
python model_xgb_5.py -max_depth 8 -n_estimators 2000 -offset 13 -random_state_cv 213376
python model_xgb_5.py -max_depth 9 -n_estimators 2000 -offset 17 -random_state_cv 213386

python model_erf_1.py -criterion entropy -n_estimators 200 -random_state_cv 213396
python model_erf_1.py -criterion entropy -n_estimators 500 -random_state_cv 231406
python model_erf_1.py -criterion gini -n_estimators 200 -random_state_cv 214136
python model_erf_1.py -criterion gini -n_estimators 500 -random_state_cv 663

python model_erf_2.py -criterion entropy -n_estimators 200 -max_categories 10 -random_state_cv 213426
python model_erf_2.py -criterion entropy -n_estimators 500 -max_categories 20 -random_state_cv 231436
python model_erf_2.py -criterion gini -n_estimators 200 -max_categories 15 -random_state_cv 213446
python model_erf_2.py -criterion gini -n_estimators 500 -max_categories 25 -random_state_cv 214356

python model_rf_2.py -criterion entropy -n_estimators 200 -max_categories 10 -random_state_cv 214636
python model_rf_2.py -criterion entropy -n_estimators 500 -max_categories 25 -random_state_cv 214736
python model_rf_2.py -criterion gini -n_estimators 200 -max_categories 15 -random_state_cv 214863
python model_rf_2.py -criterion gini -n_estimators 500 -max_categories 25 -random_state_cv 214396

python model_xgb_6.py -max_depth 7 -n_estimators 300 -offset 7 -random_state_cv 215306
python model_xgb_6.py -max_depth 8 -n_estimators 400 -offset 23 -random_state_cv 215316
python model_xgb_6.py -max_depth 9 -n_estimators 500 -offset 29 -random_state_cv 215326

python model_xgb_1_sub.py -max_depth 6 -n_estimators 200 -random_state_cv 215346
python model_xgb_1_sub.py -max_depth 8 -n_estimators 200 -random_state_cv 215464
python model_xgb_1_sub.py -max_depth 20 -n_estimators 200 -random_state_cv 214556


python model_erf_3.py -criterion entropy -n_estimators 200 -max_categories 13 -random_state_cv 215466
python model_erf_3.py -criterion entropy -n_estimators 500 -max_categories 23 -random_state_cv 214576
python model_erf_3.py -criterion gini -n_estimators 200 -max_categories 18 -random_state_cv 215846
python model_erf_3.py -criterion gini -n_estimators 500 -max_categories 28 -random_state_cv 214596

python model_rf_3.py -criterion entropy -n_estimators 200 -random_state_cv 216064
python model_rf_3.py -criterion entropy -n_estimators 500 -random_state_cv 214616
python model_rf_3.py -criterion gini -n_estimators 200 -random_state_cv 216264
python model_rf_3.py -criterion gini -n_estimators 500 -random_state_cv 216364

python model_sgd_1.py -max_categories 20 -random_state_cv 216464
python model_sgd_1.py -max_categories 60 -random_state_cv 216564
python model_sgd_2.py -max_categories 80 -random_state_cv 216664

python model_ohnb.py -var_name v22 -random_state_cv 216476
python model_ohnb.py -var_name v72 -random_state_cv 2164846
python model_ohnb.py -var_name v38 -random_state_cv 216946
python model_ohnb.py -var_name v113 -random_state_cv 214706


python model_nn_1.py -first_layer 200 -second_layer 10 -n_epochs 100 -random_state_cv 214816
python model_nn_1.py -first_layer 400 -second_layer 50 -n_epochs 100 -random_state_cv 218426
python model_nn_1.py -first_layer 100 -second_layer 200 -n_epochs 100 -random_state_cv 218436
python model_nn_1.py -first_layer 400 -second_layer 100 -n_epochs 100 -random_state_cv 218446

python model_nn_2.py -first_layer 128 -second_layer 64 -third_layer 32 -n_epochs 120 -random_state_cv 241856
python model_nn_2.py -first_layer 64 -second_layer 64 -third_layer 64 -n_epochs 150 -random_state_cv 218466

python model_ohnb_interaction.py -var_name1 v66 -var_name2 v113 -random_state_cv 218746
python model_serf_1.py -max_depth 20 -max_features 50 -n_estimators 800 -random_state_cv 214886

python model_nn_2.py -first_layer 64 -second_layer 32 -third_layer 64 -n_epochs 150 -random_state_cv 241896
python model_nn_2.py -first_layer 128 -second_layer 64 -third_layer 64 -n_epochs 150 -random_state_cv 219046
python model_nn_2.py -first_layer 256 -second_layer 128 -third_layer 64 -n_epochs 150 -random_state_cv 219416

python feature_forwarding.py -feature_name v504
python feature_forwarding.py -feature_name v104
python feature_forwarding.py -feature_name v124
python feature_forwarding.py -feature_name v214
python feature_forwarding.py -feature_name v1144


python model_dt_1.py -criterion entropy -max_depth 5 -random_state_cv 219264
python model_dt_1.py -criterion entropy -max_depth 3 -random_state_cv 219364

python model_dt_1.py -criterion gini -max_depth 5 -random_state_cv 219464
python model_dt_1.py -criterion gini -max_depth 3 -random_state_cv 219564


del *.pyc
del .Rhistory

pause