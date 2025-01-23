import torch
import torch.nn as nn
import torch.nn.functional as F


class WindPowerFFNN(nn.Module):
    def __init__(self, nwp_input_size:int, with_px:bool=False,):
        input_dim = nwp_input_size*48
        super(WindPowerFFNN, self).__init__()
        self.with_px = with_px
        if with_px:
            self.px_proj = nn.Linear(96, 48)
            input_dim += 48
        self.fc = nn.Linear(input_dim, 96)
        self.sigmoid = nn.Sigmoid()


    def forward(self, nwp_data, px=None):
        # px.shape: (batch_size, 96, 1)
        if self.with_px:
            px = self.px_proj(px.squeeze(-1)).unsqueeze(-1)
        x = torch.cat((nwp_data, px), dim=-1)
        
        x = x.reshape(x.shape[0], -1)
        output = self.fc(x)  # [B, 96]
        output = self.sigmoid(output)
        return output


class EnhancedWindPowerNN(nn.Module):
    def __init__(self, nwp_input_size:int, with_px:bool=False,):
        super(EnhancedWindPowerNN, self).__init__()
        input_dim = nwp_input_size*48
        self.with_px = with_px
        if with_px:
            self.px_proj = nn.Linear(96, 48)
            input_dim += 48
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 96)
        self.dropout = nn.Dropout(0.3)

    def forward(self, nwp_data, px=None):
        # px.shape: (batch_size, 96, 1)
        if self.with_px:
            px = self.px_proj(px.squeeze(-1)).unsqueeze(-1)
        x = torch.cat((nwp_data, px), dim=-1)
        
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x