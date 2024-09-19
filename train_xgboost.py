import argparse
import pickle
from lightgbm import LGBMRegressor
from dataloading import get_dataset_and_denormalizer_sklearn
from draw import plot_predictions_vs_ground_truth_vanilla
from utils import compute_all_metrics, write_csv
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
_, Y_train, X_nwp_train, denormalizer = get_dataset_and_denormalizer_sklearn(args.plant_number, "train", str(save_path))
model.fit(X_nwp_train, Y_train)
with open(save_path.get_model_path(), "wb") as f:
    pickle.dump(model,f)
_, Y_test, X_nwp_test, _ = get_dataset_and_denormalizer_sklearn(args.plant_number, "test", str(save_path))
preds_test = model.predict(X_nwp_test)
preds_test = denormalizer(preds_test)
Y_test = denormalizer(Y_test)
all_metrics = compute_all_metrics(preds_test, Y_test, denormalizer(1.0))
print(all_metrics)
write_csv(save_path.get_metrics_path(), all_metrics)
plot_predictions_vs_ground_truth_vanilla(preds_test, Y_test, save_path.get_png_path())