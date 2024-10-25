import torch
import torch.nn as nn
from models.encoders import SpectralEncoder, TemporalEncoder, SpatialEncoder
from models.attention import MultimodalAttention
from models.diffusion import DiffusionProcess

class DiffusionTransformer(nn.Module):
    def __init__(self, input_dim_spectral, input_dim_temporal, input_dim_spatial, embed_dim, timesteps):
        super(DiffusionTransformer, self).__init__()
        # Encoders
        self.spectral_encoder = SpectralEncoder(input_dim=input_dim_spectral, embed_dim=embed_dim)
        self.temporal_encoder = TemporalEncoder(input_dim=input_dim_temporal, embed_dim=embed_dim)
        self.spatial_encoder = SpatialEncoder(input_dim=input_dim_spatial, embed_dim=embed_dim)
        
        # Attention
        self.attn = MultimodalAttention(embed_dim=embed_dim)

        # Diffusion process
        self.diffusion = DiffusionProcess(timesteps=timesteps)
        
        # Classifier
        self.fc = nn.Linear(embed_dim, 10)  # Assuming 10 classes

    def forward(self, x_spectral, x_temporal, x_spatial, t):
        # Encode each modality
        x_spectral = self.spectral_encoder(x_spectral)
        x_temporal = self.temporal_encoder(x_temporal)
        x_spatial = self.spatial_encoder(x_spatial)

        # Combine encoded modalities using attention
        combined_features, _ = self.attn(x_spectral, x_temporal, x_spatial)

        # Apply diffusion process (forward)
        x_diffused, _ = self.diffusion.forward_diffusion(combined_features, t)

        # Classification head
        output = self.fc(x_diffused.mean(dim=1))  # Global average pooling before classification
        return output





