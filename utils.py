import csv
import numpy as np
from constants import CNN_LSTM, CROSS_VIVIT, FFNN, GDBOOST, GPNN, GREEK, LSTM, XGBOOST, sklearn_model_type_dict
from dataloading import get_data_loaders_and_denormalizer
from lstm_model import BiLSTMNWPOnly, CNNLSTMModel
from ffnn_model import EnhancedWindPowerNN, WindPowerFFNN
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from crossvivit_model import RoCrossViViT
from metrics import CR, MAE, compute_gte, compute_pte, time_delay_error


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def get_model_and_loader(args, device):
    if args.model_type == LSTM:
        # Get data loaders
        train_loader, val_loader, test_loader, denormalizer = get_data_loaders_and_denormalizer(args.plant_number, args.batch_size, True)
        # Initialize model, criterion, and optimizer
        model = BiLSTMNWPOnly().to(device)
    elif args.model_type == CNN_LSTM:
        # Get data loaders
        train_loader, val_loader, test_loader, denormalizer = get_data_loaders_and_denormalizer(args.plant_number, args.batch_size, False)
        # Initialize model, criterion, and optimizer
        model = CNNLSTMModel().to(device)
    elif args.model_type == FFNN:
        # Get data loaders
        train_loader, val_loader, test_loader, denormalizer = get_data_loaders_and_denormalizer(args.plant_number, args.batch_size, False)
        # Initialize model, criterion, and optimizer
        model = WindPowerFFNN().to(device)
    elif args.model_type == GPNN:
        # Get data loaders
        train_loader, val_loader, test_loader, denormalizer = get_data_loaders_and_denormalizer(args.plant_number, args.batch_size, False)
        # Initialize model, criterion, and optimizer
        model = EnhancedWindPowerNN().to(device)
    elif args.model_type == CROSS_VIVIT:
        train_loader, val_loader, test_loader, denormalizer = get_data_loaders_and_denormalizer(args.plant_number, args.batch_size, False)
        model = RoCrossViViT().to(device)
    return model, train_loader, val_loader, test_loader, denormalizer


def get_sklearn_model(model_type_int:int):
    model_type = sklearn_model_type_dict[model_type_int]
    if model_type == XGBOOST:
        model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1, max_depth=5, alpha=10, n_estimators=100)
    elif model_type == GDBOOST:
        model = MultiOutputRegressor(GradientBoostingRegressor(loss='squared_error', learning_rate=0.1, max_depth=5, alpha=0.1, n_estimators=10, random_state=42))
    elif model_type == GREEK:
        model = MultiOutputRegressor(ExtraTreesRegressor(n_estimators=100, random_state=42))
    return model


def compute_all_metrics(preds, gts, cap):
    if isinstance(preds, list):
        preds = np.concatenate(preds, axis=0).flatten()
    if isinstance(gts, list):
        gts = np.concatenate(gts, axis=0).flatten()
    rmse = CR(preds, gts, cap)*100
    mae = MAE(preds, gts, cap)*100
    gte = compute_gte(gts, preds)
    pte, extrema_indices = compute_pte(gts, preds)
    # tde = time_delay_error(gts, preds)
    correlation_matrix = np.corrcoef(preds, gts)
    r = correlation_matrix[0, 1]
    return {
        "rmse": rmse,
        "mae": mae,
        "gte": gte,
        "pte": pte,
        # "tde": tde,
        "r": r,
    }


def write_csv(csv_filename, all_metrics):
    with open(csv_filename, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        # 写入表头（字典的键）
        csv_writer.writerow(all_metrics.keys())
        # 写入内容（字典的值）
        csv_writer.writerow(all_metrics.values())