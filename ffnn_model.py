import torch.nn as nn
import torch.nn.functional as F


nwp_input_size = 16  # NWP data has 16 features

class WindPowerFFNN(nn.Module):
    def __init__(self, input_dim=nwp_input_size*48):
        super(WindPowerFFNN, self).__init__()
        self.fc = nn.Linear(input_dim, 96)
        self.sigmoid = nn.Sigmoid()
        # Batch Norm Layer, without which the model fails to perform
        self.batch_norm = nn.BatchNorm1d(nwp_input_size)


    def forward(self, nwp_data):
        # Apply batch normalization across the time dimension
        nwp_data = nwp_data.permute(0, 2, 1)  # Switch batch and feature dimensions
        nwp_data = self.batch_norm(nwp_data)
        nwp_data = nwp_data.permute(0, 2, 1)  # Switch them back to [B, 96, 16]
        x = nwp_data.reshape(nwp_data.shape[0], -1)
        output = self.fc(x)  # [B, 96]
        output = self.sigmoid(output)
        return output


class EnhancedWindPowerNN(nn.Module):
    def __init__(self, input_dim=nwp_input_size*48):
        super(EnhancedWindPowerNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 96)
        self.dropout = nn.Dropout(0.3)

    def forward(self, nwp_data):
        x = nwp_data.reshape(nwp_data.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x