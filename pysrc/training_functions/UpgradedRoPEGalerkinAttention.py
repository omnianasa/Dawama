import torch
import torch.nn as nn
from training_functions.RandomFourierEmbedding import RandomFourierEmbedding

class UpgradedRoPEGalerkinAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)
        
    def apply_2d_rope(self, q, k, queries_coords):
        q_x, q_y = q[..., :self.d_model//2], q[..., self.d_model//2:]
        k_x, k_y = k[..., :self.d_model//2], k[..., self.d_model//2:]
        
        x = queries_coords[:, 0].unsqueeze(1).unsqueeze(2)
        y = queries_coords[:, 1].unsqueeze(1).unsqueeze(2)
        
        q_x_rot = q_x * torch.cos(x) - q_y * torch.sin(x)
        q_y_rot = q_x * torch.sin(y) + q_y * torch.cos(y)
        q_roped = torch.cat([q_x_rot, q_y_rot], dim=-1)
        
        k_x_rot = k_x * torch.cos(x) - k_y * torch.sin(x)
        k_y_rot = k_x * torch.sin(y) + k_y * torch.cos(y)
        k_roped = torch.cat([k_x_rot, k_y_rot], dim=-1)
        
        return q_roped, k_roped

    def forward(self, q_input, kv_input, queries_coords):
        Q = self.q_linear(q_input)   
        K = self.k_linear(kv_input)  
        V = self.v_linear(kv_input)  
        
        Q, K = self.apply_2d_rope(Q, K, queries_coords)
        
        Q = Q / (torch.norm(Q, p=2, dim=1, keepdim=True) + 1e-6)
        K = K / (torch.norm(K, p=2, dim=1, keepdim=True) + 1e-6)
        V = V / (torch.norm(V, p=2, dim=1, keepdim=True) + 1e-6)
        
        n_sensors = K.shape[1]
        kv_prod = torch.matmul(K.transpose(1, 2), V) / n_sensors 
        attn_out = torch.matmul(Q, kv_prod) 

        return self.norm(attn_out)
        