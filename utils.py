from constants import FFNN, LSTM
from dataloading import get_data_loaders_and_denormalizer
from lstm_model import BiLSTMNWPOnly
from ffnn_model import WindPowerFFNN


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