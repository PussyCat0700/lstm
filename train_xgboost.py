import argparse
import os
import pickle
from lightgbm import LGBMRegressor
from dataloading import get_dataset_and_denormalizer_sklearn
from paths import XGBSavePath


parser = argparse.ArgumentParser()
parser.add_argument("plant_number", type=int, help="Power plant number to be used for training")
args = parser.parse_args()

params_LGBM_wind_trading={
    'objective':'mse',
    'num_leaves': 1000,
    'n_estimators': 500,
    'max_depth':6,
    'min_data_in_leaf': 700,
    'learning_rate':0.078,
    'lambda_l1': 70,
    'lambda_l2': 40,
    'verbose':-1,
}
save_path = XGBSavePath(args.plant_number)

model=LGBMRegressor(**params_LGBM_wind_trading)
X_train, Y_label, X_nwp, denormalize_power_data = get_dataset_and_denormalizer_sklearn(0, "train")
model.fit(X_nwp, Y_label)
with open(save_path.get_model_path(), "wb") as f:
    pickle.dump(model,f)