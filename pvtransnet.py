from crossvivit_model import Transformer
import torch.nn as nn
from paths import nwp_input_size


class PVTransNetE(nn.Module):
    def __init__(self):
        super().__init__()
        dim = 128
        self.num_mlp_heads = 2
        self.embedding = nn.Linear(nwp_input_size, dim)
        self.model = Transformer(
            dim,
            48,
            1,
            2,
            128,
            512,
            dropout=0.1,
        )
        self.fc1 = nn.Linear(dim * 48, 256)  # 修改线性层输入大小
        self.fc2 = nn.Linear(256, 96)
    
    def forward(self, x):
        B = x.shape[0]
        emb = self.embedding(x)
        y = self.model(emb)
        out = self.fc1(y.reshape(B, -1))
        out = self.fc2(out)
        return out