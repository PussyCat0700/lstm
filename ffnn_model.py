import torch.nn as nn


nwp_input_size = 16  # NWP data has 16 features

class WindPowerFFNN(nn.Module):
    def __init__(self, input_dim=nwp_input_size, output_dim=1):
        super(WindPowerFFNN, self).__init__()
        self.nwp_mlp = nn.Linear(48, 96)
        self.fc = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        # Batch Norm Layer, without which the model fails to perform
        self.batch_norm = nn.BatchNorm1d(nwp_input_size)


    def forward(self, nwp_data):
        nwp_norm = (nwp_data - nwp_data.min()) / (nwp_data.max() - nwp_data.min())
        nwp_data = self.nwp_mlp(nwp_norm.transpose(-1, -2)).transpose(-1, -2)  # [B, 96, nwp_input_size]
        # Apply batch normalization across the time dimension
        nwp_data = nwp_data.permute(0, 2, 1)  # Switch batch and feature dimensions
        nwp_data = self.batch_norm(nwp_data)
        nwp_data = nwp_data.permute(0, 2, 1)  # Switch them back
        output = self.fc(nwp_data)  # [B, 96, output_dim]
        output = self.sigmoid(output).squeeze(-1)
        return output