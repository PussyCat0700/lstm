import torch
import torch.nn as nn
import torch.optim as optim


# Hyperparameters
nwp_input_size = 16  # NWP data has 16 features
power_input_size = 1  # Power data is single-channel, so input size is 1
hidden_size = 100
output_size = 96  # Predicting the next 24 hours, 15 minutes per unit
num_layers = 2

class BiLSTMWithFusion(nn.Module):
    def __init__(self,
                 input_dim=nwp_input_size+power_input_size,
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
        # Concatenate the power and NWP data along the feature dimension
        nwp_norm = (nwp_data - nwp_data.min()) / (nwp_data.max() - nwp_data.min())
        nwp_data = self.nwp_mlp(nwp_norm.transpose(-1, -2)).transpose(-1, -2)
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

# Initialize model, loss function, and optimizer
# model = BiLSTMWithFusion(nwp_input_size=nwp_input_size, power_input_size=power_input_size,
#                          hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# for epoch in range(num_epochs):
#     for X, Y, nwp_data in train_loader:
#         # Reshape power data to add feature dimension (96,1) for LSTM input
#         power_data = X.view(X.size(0), X.size(1), 1)  
#         output = model(nwp_data, power_data)
#         loss = criterion(output, Y)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
