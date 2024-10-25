import torch
import torch.nn as nn

class SpectralEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_layers=6):
        super(SpectralEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8), 
            num_layers=num_layers
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return x

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_layers=6):
        super(TemporalEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8), 
            num_layers=num_layers
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return x

class SpatialEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_layers=6):
        super(SpatialEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8), 
            num_layers=num_layers
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return x




