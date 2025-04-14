import torch
from crossvivit_model import Transformer
import torch.nn as nn


class PVTransNetE(nn.Module):
    def __init__(self, input_dim, nwp_input_len:int, with_px:bool=False):
        super().__init__()
        dim = 128
        self.num_mlp_heads = 2
        self.with_px = with_px
        if with_px:
            self.px_proj = nn.Linear(96, nwp_input_len)
            input_dim+=1
        self.embedding = nn.Linear(input_dim, dim)
        self.model = Transformer(
            dim,
            nwp_input_len,
            1,
            2,
            128,
            512,
            dropout=0.1,
        )
        self.fc1 = nn.Linear(dim * nwp_input_len, 256)  # 修改线性层输入大小
        self.fc2 = nn.Linear(256, 96)
    
    def forward(self, x, px=None):
        B = x.shape[0]
        # px.shape: (batch_size, 96, 1)
        if self.with_px:
            px = self.px_proj(px.squeeze(-1)).unsqueeze(-1)
            x = torch.cat((x, px), dim=-1)
        emb = self.embedding(x)
        y = self.model(emb)
        out = self.fc1(y.reshape(B, -1))
        out = self.fc2(out)
        return out