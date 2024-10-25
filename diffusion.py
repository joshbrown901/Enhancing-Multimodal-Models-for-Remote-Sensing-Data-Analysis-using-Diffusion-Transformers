import torch
import torch.nn as nn
import math

class DiffusionProcess(nn.Module):
    def __init__(self, timesteps, beta_min=0.0001, beta_max=0.02):
        super(DiffusionProcess, self).__init__()
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_min, beta_max, timesteps)

    def forward_diffusion(self, x, t):
        alpha_t = self.betas[t]
        noise = torch.randn_like(x)
        return torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise, noise

    def reverse_diffusion(self, x_t, t, model_output):
        alpha_t = self.betas[t]
        return (x_t - torch.sqrt(1 - alpha_t) * model_output) / torch.sqrt(alpha_t)
    




