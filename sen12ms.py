import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SEN12MSDataset(Dataset):
    def __init__(self, data_dir):
        # Load the dataset
        # Example code - actual implementation will depend on the dataset format
        self.spectral_data = np.load(data_dir + '/spectral.npy')
        self.temporal_data = np.load(data_dir + '/temporal.npy')
        self.spatial_data = np.load(data_dir + '/spatial.npy')
        self.labels = np.load(data_dir + '/labels.npy')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spectral = torch.tensor(self.spectral_data[idx], dtype=torch.float32)
        temporal = torch.tensor(self.temporal_data[idx], dtype=torch.float32)
        spatial = torch.tensor(self.spatial_data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return spectral, temporal, spatial, label





