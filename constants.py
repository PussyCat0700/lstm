FFNN = "FFNN"
GPNN = "GPNN"
CNN_LSTM = "CNN_LSTM"
CROSS_VIVIT = "CROSS_VIVIT"
PVTRANS_E = "PVTRANS_E"
XGBOOST = "XGBOOST"
XGBOOST_DR_PU = "XGBOOST_P"
RANDOM_FOREST = "RDFR"
GPCF = "GPCF"
GDBOOST = "GradientBoost"
GREEK = "Greek"
model_type_dict = {
    1: FFNN,
    2: GPNN,
    3: CNN_LSTM,
    4: CROSS_VIVIT,
    5: PVTRANS_E,
}
sklearn_model_type_dict = {
    0: XGBOOST,
    1: GPCF,
    2: GDBOOST,
    3: GREEK,
    4: XGBOOST_DR_PU,
    5: RANDOM_FOREST,
}