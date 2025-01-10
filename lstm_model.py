import torch
import torch.nn as nn


# Hyperparameters
power_input_size = 1  # Power data is single-channel, so input size is 1
hidden_size = 100

class BiLSTMWithFusion(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim=hidden_size,
                 num_layers=2,
                 output_dim=1,
                 dropout=0.3):
        super(BiLSTMWithFusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # NWP MLP mapping lengths
        self.nwp_mlp = nn.Linear(48, 96)

        # BiLSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=True)
        
        # Batch Norm Layer
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        # Activation function
        self.activation = nn.ReLU()
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, nwp_data, power_data):
        nwp_data = self.nwp_mlp(nwp_data.transpose(-1, -2)).transpose(-1, -2)
        combined_input = torch.cat((power_data, nwp_data), dim=2)
        # Repeat the first and last token
        first_token = combined_input[:, 0:1, :]  # Shape: (batch_size, 1, features)
        last_token = combined_input[:, -1:, :]   # Shape: (batch_size, 1, features)
        combined_input = torch.cat([first_token, combined_input, last_token], dim=1)
        # LSTM layer
        lstm_out, _ = self.lstm(combined_input)

        # Remove the repeated tokens after LSTM processing
        lstm_out = lstm_out[:, 1:-1, :]
        
        # Apply batch normalization across the time dimension
        lstm_out = lstm_out.permute(0, 2, 1)  # Switch batch and feature dimensions
        lstm_out = self.batch_norm(lstm_out)
        lstm_out = lstm_out.permute(0, 2, 1)  # Switch them back
        
        # Apply the fully connected layer to each timestep
        output = self.fc(lstm_out)
        
        # Apply activation function
        output = self.activation(output)
        
        return output


class CNNLSTMModel(nn.Module):
    def __init__(self,
                 input_dim,
                 num_layers=2,
                 dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)  # 池化层
        self.dropout = nn.Dropout(dropout)  # Dropout层
        self.lstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True)  # LSTM层
        self.fc1 = nn.Linear(hidden_size * 24, 256)  # 修改线性层输入大小
        self.fc2 = nn.Linear(256, 96)

    def forward(self, x):
        # x.shape: (batch_size, seq_length, input_size)
        x = x.permute(0, 2, 1)  # 改变形状为 (batch_size, seq_length, input_size)
        x = self.conv1(x)  # 经过第一个卷积层
        x = nn.ReLU()(x)   # ReLU激活
        x = self.conv2(x)  # 经过第二个卷积层
        x = nn.ReLU()(x)   # ReLU激活
        x = self.pool(x)   # 池化层
        x = x.permute(0, 2, 1)  # 改变形状为 (batch_size, seq_length, num_channels)，供 LSTM 使用

        x, _ = self.lstm(x)  # LSTM层
        x = x.reshape(x.shape[0], -1)  # 改变形状为 (batch_size, seq_length * num_channels)
        x = self.dropout(x)  # Dropout层
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class BiLSTMNWPOnly(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim=hidden_size,
                 num_layers=2,
                 output_dim=1,
                 dropout=0.3):
        super(BiLSTMNWPOnly, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # NWP MLP mapping lengths
        self.division_start = 63
        self.nwp_mlp = nn.Linear(48, 192)

        # BiLSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        
        # Activation function
        self.activation = nn.ReLU()
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, nwp_data):
        # Concatenate the power and NWP data along the feature dimension
        nwp_norm = (nwp_data - nwp_data.min()) / (nwp_data.max() - nwp_data.min())
        nwp_data = self.nwp_mlp(nwp_norm.transpose(-1, -2)).transpose(-1, -2)
        # LSTM layer
        lstm_out, _ = self.lstm(nwp_data)  # [B, 48, hs*2]
        
        # Apply the fully connected layer to each timestep
        output = self.fc(lstm_out).squeeze(-1)
        
        # Apply activation function
        output = self.activation(output)
        if self.training:
            output = output[..., :self.division_start+96]
        else:
            output = output[..., self.division_start:self.division_start+96]
        return output
