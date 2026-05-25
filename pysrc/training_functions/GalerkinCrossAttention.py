import torch
import torch.nn as nn

class GalerkinCrossAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
    def forward(self, q_input, kv_input):
        # Q shape: [batch_size, 1, d_model]
        # K, V shape: [batch_size, 8, d_model]
        Q = self.q_linear(q_input)   
        K = self.k_linear(kv_input)  
        V = self.v_linear(kv_input)  
        

        Q = Q / (torch.norm(Q, p=2, dim=1, keepdim=True) + 1e-6)
        K = K / (torch.norm(K, p=2, dim=1, keepdim=True) + 1e-6)
        V = V / (torch.norm(V, p=2, dim=1, keepdim=True) + 1e-6)
        
        n_sensors = K.shape[1] 

        kv_prod = torch.matmul(K.transpose(1, 2), V) / n_sensors 

        attn_out = torch.matmul(Q, kv_prod) 
        
        return attn_out