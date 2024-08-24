import torch.nn as nn


nwp_input_size = 16  # NWP data has 16 features

class WindPowerFFNN(nn.Module):
    def __init__(self, input_dim=nwp_input_size, hidden_dim=128, output_dim=96):
        super(WindPowerFFNN, self).__init__()
        
        # First hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.sigmoid1 = nn.Sigmoid()
        
        # Second hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid2 = nn.Sigmoid()
        
        # Third hidden layer
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid3 = nn.Sigmoid()
        
        # Output layer
        self.output = nn.Linear(48*hidden_dim, output_dim)
        
    def forward(self, x):
        # Pass through the first hidden layer
        x = self.fc1(x)
        x = self.sigmoid1(x)
        
        # Pass through the second hidden layer
        x = self.fc2(x)
        x = self.sigmoid2(x)
        
        # Pass through the third hidden layer
        x = self.fc3(x)
        x = self.sigmoid3(x)
        
        # Till now x is always of shape [B, 48, hidden_dim]
        # Output layer
        output = self.output(x.reshape(x.shape[0], -1))  # [B, -1]
        
        return output