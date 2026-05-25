import torch
import torch.nn as nn

class RandomFourierEmbedding(nn.Module):
    def __init__(self, in_features=2, d_model=64, scale=10.0):
        super().__init__()
        self.register_buffer('B', torch.randn(in_features, d_model // 2) * scale)
        self.projection = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        proj = torch.matmul(x, self.B)
        fourier_features = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.projection(fourier_features)
