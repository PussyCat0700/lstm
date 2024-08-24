import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader
from paths import source_nwp_dir, train_power_file, valid_power_file, test_power_file

class PowerPlantDataset(Dataset):
    def __init__(self, split, plant_number, power_minmax=None):
        """
        Args:
            csv_file (string): Path to the CSV file with power generation data.
            nwp_dir (string): Directory with NWP future 49-hour forecast .npy files.
            plant_number (int): Index of the power plant to be used for the dataset.
            power_minmax ([float, float]) Power min and power max for the given station only.
            If None, will be determined with current file.
        """
        if split == "train":
            csv_file = train_power_file
        elif split == "valid":
            csv_file = valid_power_file
        elif split == "test":
            csv_file = test_power_file
        self.data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        self.nwp_dir = source_nwp_dir
        self.plant_number = plant_number
        if power_minmax is None:
            self.power_minmax = [
                max(0.0, self.data.min()[self.plant_number]),
                self.data.max()[self.plant_number]]
        else:
            self.power_minmax = power_minmax
        self.power_min = self.power_minmax[0]
        self.power_max = self.power_minmax[1]

        # Ensure the plant number is valid
        if self.plant_number < 0 or self.plant_number >= self.data.shape[1]:
            raise ValueError(f"Invalid plant number: {self.plant_number}. Must be between 0 and {self.data.shape[1] - 1}.")

    def normalize_power_data(self, data):
        return (data - self.power_min) / (self.power_max - self.power_min)
    
    def denormalize_power_data(self, data):
        return data * (self.power_max - self.power_min) + self.power_min

    def __len__(self):
        return len(self.data) - 96*2  # Each sample requires data from two consecutive days. 96 points for each day

    def _get_start_time(self, idx):
        start_time = self.data.index[idx].replace(minute=0, second=0, microsecond=0)
        return start_time

    def __getitem__(self, idx):
        # Current day data
        start_time = self._get_start_time(idx)
        end_time = start_time + pd.DateOffset(hours=23, minutes=45)
        X = self.data.loc[start_time:end_time].iloc[:, self.plant_number].values
        X_norm = self.normalize_power_data(X)
        
        # Next day data
        next_start_time = start_time + pd.DateOffset(days=1)
        next_end_time = next_start_time + pd.DateOffset(hours=23, minutes=45)
        Y = self.data.loc[next_start_time:next_end_time].iloc[:, self.plant_number].values
        Y_norm = self.normalize_power_data(Y)

        # Load the corresponding NWP data
        nwp_file = os.path.join(self.nwp_dir, f"{start_time.strftime('%Y-%m-%d_%H:%M:%S')}.npy")
        nwp_data = np.load(nwp_file)

        return torch.tensor(X_norm, dtype=torch.float32), torch.tensor(Y_norm, dtype=torch.float32), torch.tensor(nwp_data[self.plant_number], dtype=torch.float32)


class PowerPlantDailyDataset(PowerPlantDataset):
    """
    1. csv不需要倒时差
    - 历史数据起始 终止
    day0					day1
    08:15:00【含】-> +96 08:00:00【含】
    - 预测数据起始 终止
    day1					day1			day2[00:00:00-23:45:00]
    +96 08:15:00【含】-> +15h45m(dropped)+24h(要的)时刻 【含】【要不就192个】
    评估：
    2. nwp的时差
    08:00:00->00:00:00 -8h
    """
    def __init__(self, split, plant_number, power_minmax=None, with_extra_span=True):
        super().__init__(split, plant_number, power_minmax)
        self.with_extra_span = with_extra_span
    
    def __len__(self):
        return len(self.data) // 96 -  2

    def _get_start_time(self, idx):
        start_time = self.data.index[idx*96].replace(hour=8, minute=15, second=0, microsecond=0)
        return start_time
    
    def __getitem__(self, idx):
        # Current day data
        start_time = self._get_start_time(idx)  # day0
        end_time = start_time + pd.DateOffset(hours=23, minutes=45)  # day1
        X = self.data.loc[start_time:end_time].iloc[:, self.plant_number].values
        X_norm = self.normalize_power_data(X)
        
        # Next day data
        if self.with_extra_span:
            # total span: 63+96=159
            next_start_time = start_time + pd.DateOffset(days=1)  # day1 start 08:15:00
            next_end_time = next_start_time + pd.DateOffset(hours=15, minutes=45) + pd.DateOffset(hours=23, minutes=45)  # day2 end 23:45:00
        else:
            # total span: 96
            next_start_time = start_time + pd.DateOffset(days=1) + pd.DateOffset(hours=15, minutes=45)  # day2 start 00:00:00
            next_end_time = next_start_time + pd.DateOffset(hours=23, minutes=45)  # day2 end 23:45:00
        Y = self.data.loc[next_start_time:next_end_time].iloc[:, self.plant_number].values
        Y_norm = self.normalize_power_data(Y)

        # Load the corresponding NWP data
        nwp_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0)
        nwp_file = os.path.join(self.nwp_dir, f"{nwp_time.strftime('%Y-%m-%d_%H:%M:%S')}.npy")
        nwp_data = np.load(nwp_file)

        return torch.tensor(X_norm, dtype=torch.float32), torch.tensor(Y_norm, dtype=torch.float32), torch.tensor(nwp_data[self.plant_number], dtype=torch.float32)


def get_data_loaders_and_denormalizer(plant_number, batch_size, with_extra_span:bool=True):
    train_dataset = PowerPlantDailyDataset("train", plant_number, with_extra_span=with_extra_span)
    power_minmax = train_dataset.power_minmax
    valid_dataset = PowerPlantDailyDataset("valid", plant_number, power_minmax, with_extra_span=False)
    test_dataset = PowerPlantDailyDataset("test", plant_number, power_minmax, with_extra_span=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.denormalize_power_data


def save_checkpoint(state, filename):
    torch.save(state, filename)


def get_latest_checkpoint(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]), reverse=True)
    return os.path.join(checkpoint_dir, checkpoints[0])


def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint.get('loss', None)
    return start_epoch, loss