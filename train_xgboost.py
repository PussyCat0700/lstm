import argparse
import pickle
import xgboost as xgb
from dataloading import get_dataset_and_denormalizer_sklearn
from draw import plot_predictions_vs_ground_truth_vanilla
from constants import XGBOOST
from utils import compute_all_metrics, write_csv
from paths import PLANTS, path_loader
import os
import traceback


parser = argparse.ArgumentParser()
parser.add_argument("plant_number", type=int, help="Power plant number to be used for training")
parser.add_argument("months", help="months used in training set.")
parser.add_argument("plant_set", choices=PLANTS.keys())
args = parser.parse_args()
path_loader.init(args.months, args.plant_set, args.plant_number)
save_path, is_done = path_loader.get_run_path_status(XGBOOST)
if not path_loader.check_exists():
    print(f"{args.plant_number} does not have source input file")
    exit(0)
if is_done:
    print(f"{args.plant_number} already has output metrics.csv at {save_path}")
    exit(0)
else:
    print(f"training in {save_path}")
model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3,
                           learning_rate=0.1, max_depth=5, alpha=10, n_estimators=100)
_, Y_train, X_nwp_train, denormalizer = get_dataset_and_denormalizer_sklearn(args.plant_number, "train", save_path)
model.fit(X_nwp_train, Y_train)
model_ckpt = os.path.join(save_path, 'model.ckpt')
with open(model_ckpt, "wb") as f:
    pickle.dump(model,f)
_, Y_test, X_nwp_test, _ = get_dataset_and_denormalizer_sklearn(args.plant_number, "test", save_path)


try:
    preds_test = model.predict(X_nwp_test)
    preds_test = denormalizer(preds_test)
    Y_test = denormalizer(Y_test)
    all_metrics = compute_all_metrics(preds_test, Y_test, denormalizer(1.0))
    print(all_metrics)
    metrics_path = os.path.join(save_path, 'metrics.csv')
    write_csv(metrics_path, all_metrics)
    png_path = os.path.join(save_path, f"{args.plant_number}.png")
    plot_predictions_vs_ground_truth_vanilla(preds_test, Y_test, png_path)
except ValueError as e:
    # 检查是否为NaN相关的ValueError
    if "Input contains NaN" in str(e):
        print("NaN-related ValueError detected.")
        
        # 创建NAN_FOUND文件
        with open(os.path.join(save_path, "NAN_FOUND"), "w") as f:
            f.write("NaN-related ValueError detected.")
            
        # 终止程序
        exit()
    else:
        # 对于其他ValueError类型，创建OTHER_FOUND文件并保存堆栈信息
        with open(os.path.join(save_path, "OTHER_FOUND"), "w") as f:
            f.write("Other ValueError detected:\n")
            f.write(traceback.format_exc())
        exit()
        
except Exception as e:
    # 捕获非ValueError的其他异常，生成OTHER_FOUND文件并记录堆栈信息
    with open(os.path.join(save_path, "OTHER_FOUND"), "w") as f:
        f.write("Non-ValueError exception detected:\n")
        f.write(traceback.format_exc())
    exit()