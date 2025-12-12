# dinov2/data/datasets/eeg_dataset.py

import torch
import numpy as np
import os
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    def __init__(self, data_root, transform=None):
        super().__init__()
        self.data_root = data_root
        self.transform = transform  # DataAugmentationEEG instance or None
        
        self.files = []
        if os.path.exists(data_root):
             self.files = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith('.npy')]
        if len(self.files) == 0:
            print(f"Warning: No .npy files found in {data_root}")

    def __len__(self):
        return len(self.files)

    def _normalize_per_channel(self, x):
        # x shape: (C, N, T)
        # Calculate mean/std along (N, T) dimensions
        mean = x.mean(axis=(1, 2), keepdims=True)
        std = x.std(axis=(1, 2), keepdims=True)
        return np.divide(x - mean, std, out=np.zeros_like(x), where=std > 1e-8)

    def __getitem__(self, index):
        path = self.files[index]
        
        # 1. Load Data
        # Raw: (C, N, T) -> (19, 30, 250)
        data_np = np.load(path)

        # 2. Per-Channel Normalization
        data = self._normalize_per_channel(data_np)
        data = torch.from_numpy(data).float()
         
        # 3. Permute for model input
        # Target: (T, C, N) -> (250, 19, 30)
        data = data.permute(2, 0, 1) 

        # 4. Apply Augmentation (Crops + Noise etc.)
        if self.transform is not None:
            # transform 返回一个 dict，包含 global_crops, local_crops, indices 等
            return self.transform(data)
        else:
            # Fallback output (should generally not happen in training)
            return {"data": data}