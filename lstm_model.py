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
                 nwp_input_size=nwp_input_size,
                 power_input_size=power_input_size,
                 hidden_size=hidden_size,
                 output_size=output_size,
                 num_layers=2):
        super(BiLSTMWithFusion, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bi-LSTM for NWP data
        self.nwp_lstm = nn.LSTM(nwp_input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        
        # Bi-LSTM for Power data
        self.power_lstm = nn.LSTM(power_input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        
        # Fully connected layer for fusion and final output
        self.fc = nn.Linear(hidden_size * 4, output_size)  # Multiply by 4 for bidirectional + fusion of two LSTMs

    def forward(self, nwp_data, power_data):
        # Initialize hidden state and cell state for NWP LSTM
        h0_nwp = torch.zeros(self.num_layers * 2, nwp_data.size(0), self.hidden_size).to(nwp_data.device)  # 2 for bidirectional
        c0_nwp = torch.zeros(self.num_layers * 2, nwp_data.size(0), self.hidden_size).to(nwp_data.device)
        
        # Initialize hidden state and cell state for Power LSTM
        h0_power = torch.zeros(self.num_layers * 2, power_data.size(0), self.hidden_size).to(power_data.device)  # 2 for bidirectional
        c0_power = torch.zeros(self.num_layers * 2, power_data.size(0), self.hidden_size).to(power_data.device)
        
        # LSTM output for NWP data
        nwp_out, _ = self.nwp_lstm(nwp_data, (h0_nwp, c0_nwp))
        
        # LSTM output for Power data
        power_out, _ = self.power_lstm(power_data, (h0_power, c0_power))
        
        # Concatenate the last time steps from both LSTM outputs
        combined = torch.cat((nwp_out[:, -1, :], power_out[:, -1, :]), dim=1)
        
        # Pass through fully connected layer to generate final output
        out = self.fc(combined)
        
        return out

# Initialize model, loss function, and optimizer
model = BiLSTMWithFusion(nwp_input_size=nwp_input_size, power_input_size=power_input_size,
                         hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
