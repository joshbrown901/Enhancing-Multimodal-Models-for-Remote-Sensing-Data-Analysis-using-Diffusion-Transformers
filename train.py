import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.model import DiffusionTransformer
from datasets.sen12ms import SEN12MSDataset
from utils import calculate_metrics

def train(model, dataloader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for spectral, temporal, spatial, labels in dataloader:
            spectral, temporal, spatial, labels = spectral.to(device), temporal.to(device), spatial.to(device), labels.to(device)
            
            # Random timestep for diffusion
            t = torch.randint(0, model.diffusion.timesteps, (1,)).to(device)
            
            # Forward pass
            outputs = model(spectral, temporal, spatial, t)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for spectral, temporal, spatial, labels in dataloader:
            spectral, temporal, spatial, labels = spectral.to(device), temporal.to(device), spatial.to(device), labels.to(device)
            
            t = torch.randint(0, model.diffusion.timesteps, (1,)).to(device)
            outputs = model(spectral, temporal, spatial, t)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics (accuracy, F1-score, IoU)
    calculate_metrics(all_labels, all_preds)





