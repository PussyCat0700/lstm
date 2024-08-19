import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader
from paths import source_nwp_dir, train_power_file, valid_power_file, test_power_file

class PowerPlantDataset(Dataset):
    def __init__(self, split, plant_number):
        """
        Args:
            csv_file (string): Path to the CSV file with power generation data.
            nwp_dir (string): Directory with NWP future 49-hour forecast .npy files.
            plant_number (int): Index of the power plant to be used for the dataset.
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

        # Ensure the plant number is valid
        if self.plant_number < 0 or self.plant_number >= self.data.shape[1]:
            raise ValueError(f"Invalid plant number: {self.plant_number}. Must be between 0 and {self.data.shape[1] - 1}.")

    def __len__(self):
        return len(self.data) - 96*2  # Each sample requires data from two consecutive days. 96 points for each day

    def __getitem__(self, idx):
        # Current day data
        start_time = self.data.index[idx].replace(minute=0, second=0, microsecond=0)
        end_time = start_time + pd.DateOffset(hours=23, minutes=45)
        X = self.data.loc[start_time:end_time].iloc[:, self.plant_number].values
        
        # Next day data
        next_start_time = start_time + pd.DateOffset(days=1)
        next_end_time = next_start_time + pd.DateOffset(hours=23, minutes=45)
        Y = self.data.loc[next_start_time:next_end_time].iloc[:, self.plant_number].values

        # Load the corresponding NWP data
        nwp_file = os.path.join(self.nwp_dir, f"{start_time.strftime('%Y-%m-%d_%H:%M:%S')}.npy")
        nwp_data = np.load(nwp_file)

        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32), torch.tensor(nwp_data[self.plant_number], dtype=torch.float32)

# Example usage
# dataset = PowerPlantDataset(csv_file='/home/yfliu/horizontal/windmodel_baselines/lstm/inputs/test_nmg_wf_history.csv', nwp_dir='/data1/yfliu/windpower_baseline/weather_history', plant_number=0)
# X, Y, nwp_data = dataset[0]


def get_data_loaders(plant_number, batch_size):
    train_dataset = PowerPlantDataset("train", plant_number)
    valid_dataset = PowerPlantDataset("valid", plant_number)
    test_dataset = PowerPlantDataset("test", plant_number)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def save_checkpoint(state, filename):
    torch.save(state, filename)