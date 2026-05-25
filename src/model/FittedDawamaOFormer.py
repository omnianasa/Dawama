import torch
import torch.nn as nn
from training_functions.RandomFourierEmbedding import RandomFourierEmbedding
from training_functions.UpgradedRoPEGalerkinAttention import UpgradedRoPEGalerkinAttention

class FittedDawamaOFormer(nn.Module):
    def __init__(self, num_sensors=8, d_model=64, nhead=4):
        super().__init__()
        self.d_model = d_model
        
        self.sensor_embed = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2,
            activation=torch.tanh, batch_first=True, norm_first=True
        )
        self.sensor_self_attn = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.latent_fusion = nn.Linear(d_model * 2, d_model)
        
        self.query_embed = RandomFourierEmbedding(in_features=2, d_model=d_model)
        self.cross_attention = UpgradedRoPEGalerkinAttention(d_model)
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
    def forward(self, sensors, queries, prev_latent=None):
        batch_size = sensors.shape[0] 
        
        s_tokens = sensors.unsqueeze(-1) 
        s_embedded = self.sensor_embed(s_tokens) 
        current_kv = self.sensor_self_attn(s_embedded) # [batch_size, 8, d_model]

        if prev_latent is None:
            prev_latent = torch.zeros(batch_size, 8, self.d_model, device=sensors.device)
        else:

            prev_latent = prev_latent[:batch_size]
            
        fused_kv = torch.cat([current_kv, prev_latent], dim=-1)
        updated_kv = self.latent_fusion(fused_kv) 
        
        q_context = self.query_embed(queries).unsqueeze(1) 
        fused_features = self.cross_attention(q_context, updated_kv, queries) 
        
        output = self.decoder(fused_features.squeeze(1)) 
        
        return output, updated_kv