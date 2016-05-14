del *.pyc
del .Rhistory

python extract_labels.py

python model_1.py -max_depth 5 -n_estimators 200 -random_state_cv 1
python model_1.py -max_depth 6 -n_estimators 200 -random_state_cv 2
python model_1.py -max_depth 7 -n_estimators 200 -random_state_cv 3
python model_1.py -max_depth 8 -n_estimators 200 -random_state_cv 4

python model_1.py -max_depth 9 -n_estimators 300 -random_state_cv 5
python model_1.py -max_depth 10 -n_estimators 300 -random_state_cv 6
python model_1.py -max_depth 15 -n_estimators 300 -random_state_cv 7

python model_xgb_2.py -max_depth 6 -n_estimators 300 -offset 1 -random_state_cv 8
python model_xgb_2.py -max_depth 8 -n_estimators 300 -offset 3 -random_state_cv 9
python model_xgb_2.py -max_depth 10 -n_estimators 300 -offset 5 -random_state_cv 10
python model_xgb_2.py -max_depth 12 -n_estimators 300 -offset 11 -random_state_cv 11

python model_rf_1.py -criterion entropy -n_estimators 200 -random_state_cv 12
python model_rf_1.py -criterion entropy -n_estimators 500 -random_state_cv 13
python model_rf_1.py -criterion gini -n_estimators 200 -random_state_cv 14
python model_rf_1.py -criterion gini -n_estimators 500 -random_state_cv 15

python model_xgb_2.py -max_depth 6 -n_estimators 300 -offset 7 -random_state_cv 16
python model_xgb_2.py -max_depth 8 -n_estimators 300 -offset 23 -random_state_cv 17
python model_xgb_2.py -max_depth 10 -n_estimators 300 -offset 29 -random_state_cv 18

python model_xgb_2.py -max_depth 6 -n_estimators 300 -offset 31 -random_state_cv 19
python model_xgb_2.py -max_depth 8 -n_estimators 300 -offset 37 -random_state_cv 20
python model_xgb_2.py -max_depth 10 -n_estimators 300 -offset 39 -random_state_cv 21


python model_xgb_2.py -max_depth 6 -n_estimators 300 -offset 7 -random_state_cv 22
python model_xgb_2.py -max_depth 8 -n_estimators 300 -offset 23 -random_state_cv 23
python model_xgb_2.py -max_depth 10 -n_estimators 300 -offset 29 -random_state_cv 24

python model_xgb_2.py -max_depth 6 -n_estimators 300 -offset 31 -random_state_cv 25
python model_xgb_2.py -max_depth 8 -n_estimators 300 -offset 37 -random_state_cv 26
python model_xgb_2.py -max_depth 10 -n_estimators 300 -offset 39 -random_state_cv 27

python model_xgb_4.py -n_estimators 300 -max_depth 8 -random_state_cv 34
python model_xgb_4.py -n_estimators 500 -max_depth 8 -random_state_cv 35

python model_xgb_5.py -max_depth 7 -n_estimators 2000 -offset 2 -random_state_cv 36
python model_xgb_5.py -max_depth 8 -n_estimators 2000 -offset 13 -random_state_cv 37
python model_xgb_5.py -max_depth 9 -n_estimators 2000 -offset 17 -random_state_cv 38

python model_erf_1.py -criterion entropy -n_estimators 200 -random_state_cv 39
python model_erf_1.py -criterion entropy -n_estimators 500 -random_state_cv 40
python model_erf_1.py -criterion gini -n_estimators 200 -random_state_cv 41
python model_erf_1.py -criterion gini -n_estimators 500 

python model_erf_2.py -criterion entropy -n_estimators 200 -max_categories 10 -random_state_cv 42
python model_erf_2.py -criterion entropy -n_estimators 500 -max_categories 20 -random_state_cv 43
python model_erf_2.py -criterion gini -n_estimators 200 -max_categories 15 -random_state_cv 44
python model_erf_2.py -criterion gini -n_estimators 500 -max_categories 25 -random_state_cv 45

python model_rf_2.py -criterion entropy -n_estimators 200 -max_categories 10 -random_state_cv 46
python model_rf_2.py -criterion entropy -n_estimators 500 -max_categories 25 -random_state_cv 47
python model_rf_2.py -criterion gini -n_estimators 200 -max_categories 15 -random_state_cv 48
python model_rf_2.py -criterion gini -n_estimators 500 -max_categories 25 -random_state_cv 49

python model_xgb_6.py -max_depth 7 -n_estimators 300 -offset 7 -random_state_cv 50
python model_xgb_6.py -max_depth 8 -n_estimators 400 -offset 23 -random_state_cv 51
python model_xgb_6.py -max_depth 9 -n_estimators 500 -offset 29 -random_state_cv 52

python model_xgb_1_sub.py -max_depth 6 -n_estimators 200 -random_state_cv 53
python model_xgb_1_sub.py -max_depth 8 -n_estimators 200 -random_state_cv 54
python model_xgb_1_sub.py -max_depth 20 -n_estimators 200 -random_state_cv 55


python model_erf_3.py -criterion entropy -n_estimators 200 -max_categories 13 -random_state_cv 56
python model_erf_3.py -criterion entropy -n_estimators 500 -max_categories 23 -random_state_cv 57
python model_erf_3.py -criterion gini -n_estimators 200 -max_categories 18 -random_state_cv 58
python model_erf_3.py -criterion gini -n_estimators 500 -max_categories 28 -random_state_cv 59

python model_rf_3.py -criterion entropy -n_estimators 200 -random_state_cv 60 
python model_rf_3.py -criterion entropy -n_estimators 500 -random_state_cv 61
python model_rf_3.py -criterion gini -n_estimators 200 -random_state_cv 62
python model_rf_3.py -criterion gini -n_estimators 500 -random_state_cv 63

python model_sgd_1.py -max_categories 20 -random_state_cv 64
python model_sgd_1.py -max_categories 60 -random_state_cv 65
python model_sgd_2.py -max_categories 80 -random_state_cv 66

python model_ohnb.py -var_name v22 -random_state_cv 67
python model_ohnb.py -var_name v72 -random_state_cv 68
python model_ohnb.py -var_name v38 -random_state_cv 69
python model_ohnb.py -var_name v113 -random_state_cv 70


python model_nn_1.py -first_layer 200 -second_layer 10 -n_epochs 100 -random_state_cv 81
python model_nn_1.py -first_layer 400 -second_layer 50 -n_epochs 100 -random_state_cv 82
python model_nn_1.py -first_layer 100 -second_layer 200 -n_epochs 100 -random_state_cv 83
python model_nn_1.py -first_layer 400 -second_layer 100 -n_epochs 100 -random_state_cv 84

python model_nn_2.py -first_layer 128 -second_layer 64 -third_layer 32 -n_epochs 120 -random_state_cv 85
python model_nn_2.py -first_layer 64 -second_layer 64 -third_layer 64 -n_epochs 150 -random_state_cv 86

python model_ohnb_interaction.py -var_name1 v66 -var_name2 v113 -random_state_cv 87
python model_serf_1.py -max_depth 20 -max_features 50 -n_estimators 800 -random_state_cv 88

python model_nn_2.py -first_layer 64 -second_layer 32 -third_layer 64 -n_epochs 150 -random_state_cv 89
python model_nn_2.py -first_layer 128 -second_layer 64 -third_layer 64 -n_epochs 150 -random_state_cv 90
python model_nn_2.py -first_layer 256 -second_layer 128 -third_layer 64 -n_epochs 150 -random_state_cv 91

python feature_forwarding.py -feature_name v50
python feature_forwarding.py -feature_name v10
python feature_forwarding.py -feature_name v12
python feature_forwarding.py -feature_name v21
python feature_forwarding.py -feature_name v114


python model_dt_1.py -criterion entropy -max_depth 5 -random_state_cv 92
python model_dt_1.py -criterion entropy -max_depth 3 -random_state_cv 93

python model_dt_1.py -criterion gini -max_depth 5 -random_state_cv 94
python model_dt_1.py -criterion gini -max_depth 3 -random_state_cv 95


del *.pyc
del .Rhistory

pause