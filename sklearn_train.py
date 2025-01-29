import argparse
import pickle
import numpy as np
from dataloading import get_dataset_and_denormalizer_sklearn
from draw import plot_predictions_vs_ground_truth_vanilla
from constants import sklearn_model_type_dict
from utils import compute_all_metrics, get_sklearn_model, write_csv
from paths import KEY_NORM_NWP, KEY_NORM_Y, KEY_REAL_Y, KEY_TIME_Y, PLANTS, path_loader
import os
import traceback


def everything(args, period:int):
    path_loader.init(args.months, args.plant_set, args.plant_number, args.plant_type, period)
    modelname = sklearn_model_type_dict[args.model_type]
    save_path, is_done = path_loader.get_run_path_status(modelname)
    print(f"ckpt: {save_path}")
    if not path_loader.check_exists():
        print(f"{args.plant_number} does not have source input file")
        exit(0)
    output_path = os.path.join(save_path, f'output{period}h.csv')
    is_done = is_done and os.path.exists(output_path)
    if is_done:
        print(f"{args.plant_number} already has output metrics.csv at {save_path}")
        exit(0)
    model_ckpt = os.path.join(save_path, f'model_{period}h.ckpt')
    train_data, denormalizer = get_dataset_and_denormalizer_sklearn(args.plant_number, "train", save_path, period)
    if os.path.exists(model_ckpt):
        print(f"loading model from {model_ckpt}")
        with open(model_ckpt, 'rb') as f:
            model = pickle.load(f)
    else:
        model = get_sklearn_model(args.model_type)
        X_nwp_train = train_data[KEY_NORM_NWP]
        Y_train = train_data[KEY_NORM_Y]
        model.fit(X_nwp_train, Y_train)
        with open(model_ckpt, "wb") as f:
            pickle.dump(model,f)
    test_data, _ = get_dataset_and_denormalizer_sklearn(args.plant_number, "test", save_path, period)
    X_nwp_test = test_data[KEY_NORM_NWP]
    Y_test_real = test_data[KEY_REAL_Y]
    Y_time = test_data[KEY_TIME_Y]


    try:
        preds_test = model.predict(X_nwp_test)
        preds_test = denormalizer(preds_test)
        preds_test = preds_test.flatten()
        Y_test_real = Y_test_real.flatten()
        preds_test = np.maximum(preds_test, 0)
        Y_test_real = np.maximum(Y_test_real, 0)
        png_path = os.path.join(save_path, f"{args.plant_number}_{period}h.png")
        plot_predictions_vs_ground_truth_vanilla(preds_test, Y_test_real, png_path, days=10, all_y_times=Y_time)
        all_metrics = compute_all_metrics(preds_test, Y_test_real, denormalizer(1.0))
        print(all_metrics)
        metrics_path = os.path.join(save_path, f'metrics_{period}h.csv')
        write_csv(metrics_path, all_metrics)
    except ValueError as e:
        # 检查是否为NaN相关的ValueError
        if "Input contains NaN" in str(e):
            print("NaN-related ValueError detected.")
            
            # 创建NAN_FOUND文件
            with open(os.path.join(save_path, f"NAN_FOUND_{period}h"), "w") as f:
                f.write("NaN-related ValueError detected.")
                
            # 终止程序
            exit()
        else:
            # 对于其他ValueError类型，创建OTHER_FOUND文件并保存堆栈信息
            with open(os.path.join(save_path, f"OTHER_FOUND_{period}h"), "w") as f:
                f.write("Other ValueError detected:\n")
                f.write(traceback.format_exc())
            exit()
            
    except Exception as e:
        # 捕获非ValueError的其他异常，生成OTHER_FOUND文件并记录堆栈信息
        with open(os.path.join(save_path, f"OTHER_FOUND_{period}h"), "w") as f:
            f.write("Non-ValueError exception detected:\n")
            f.write(traceback.format_exc())
        exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", type=int)
    parser.add_argument("--plant_number", type=int, help="Power plant number to be used for training")
    parser.add_argument("--months", help="months used in training set.")
    parser.add_argument("--plant_set", choices=PLANTS.keys())
    parser.add_argument("--plant_type", type=int, choices=[0, 1], default=None, help="0 for windpower, 1 for solarpower.")
    parser.add_argument("--period", type=int, default=24)
    args = parser.parse_args()
    if args.period > 24:
        everything(args, args.period)
    else:
        for period in [1, 4, 24]:
            print(f"{period=}")
            everything(args, period)