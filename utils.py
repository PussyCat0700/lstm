import numpy as np
from constants import FFNN, LSTM
from dataloading import get_data_loaders_and_denormalizer
from lstm_model import BiLSTMNWPOnly
from ffnn_model import WindPowerFFNN
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
    elif args.model_type == FFNN:
        # Get data loaders
        train_loader, val_loader, test_loader, denormalizer = get_data_loaders_and_denormalizer(args.plant_number, args.batch_size, False)
        # Initialize model, criterion, and optimizer
        model = WindPowerFFNN().to(device)
    return model, train_loader, val_loader, test_loader, denormalizer


def compute_all_metrics(preds, gts, cap):
    preds = np.concatenate(preds, axis=0).flatten()
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