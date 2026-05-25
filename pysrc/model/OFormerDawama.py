import torch
import torch.nn as nn
from training_functions.RandomFourierEmbedding import RandomFourierEmbedding
from training_functions.GalerkinCrossAttention import GalerkinCrossAttention

class OFormerDawama(nn.Module):
    def __init__(self, num_sensors=8, d_model=64, nhead=4):
        super().__init__()

        self.sensor_embed = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2,
            activation=torch.tanh, 
            batch_first=True, norm_first=True
        )
        self.sensor_self_attn = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.query_embed = RandomFourierEmbedding(in_features=2, d_model=d_model)
        
        self.cross_attention = GalerkinCrossAttention(d_model)
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
    def forward(self, sensors, queries):
        s_tokens = sensors.unsqueeze(-1) 
        s_embedded = self.sensor_embed(s_tokens) 
        kv_context = self.sensor_self_attn(s_embedded) 
        q_context = self.query_embed(queries).unsqueeze(1) 
        fused_features = self.cross_attention(q_context, kv_context) 
        
        output = self.decoder(fused_features.squeeze(1)) 
        return output