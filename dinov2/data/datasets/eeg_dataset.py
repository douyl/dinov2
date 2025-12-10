# dinov2/data/datasets/eeg_dataset.py

import torch
import numpy as np
import os
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    def __init__(self, data_root, num_channels=19, num_patches_per_channel=30, patch_time_dim=250, 
                 local_crops_number=8, local_crop_size_channels=10, local_crop_size_patches=15):
        super().__init__()
        self.data_root = data_root
        self.C = num_channels
        self.N = num_patches_per_channel
        self.T = patch_time_dim
        
        # Hyperparameters for crops
        self.n_local_crops = local_crops_number
        self.local_C = local_crop_size_channels
        self.local_N = local_crop_size_patches
        
        self.files = []
        if os.path.exists(data_root):
             self.files = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith('.npy')]
        if len(self.files) == 0:
            print(f"Warning: No .npy files found in {data_root}")

    def __len__(self):
        return len(self.files)

    def _normalize_per_channel(self, x):
        # x shape: (C, N, T)
        # Calculate mean/std along (N, T) dimensions for each channel
        # keepdim=True results in shape (C, 1, 1)
        mean = x.mean(axis=(1, 2), keepdims=True)
        std = x.std(axis=(1, 2), keepdims=True)
        return np.divide(x - mean, std, out=np.zeros_like(x), where=std > 1e-8)

    def __getitem__(self, index):
        path = self.files[index]
        # try:
        # 1. Load Data
        # Raw: (C, N, T) -> (19, 30, 250)
        data_np = np.load(path)

        # 2. Per-Channel Normalization
        data = self._normalize_per_channel(data_np)

        data = torch.from_numpy(data).float()
         
        # 3. Permute for model input
        # Target: (T, C, N) -> (250, 19, 30) 
        # Note: dinov2 patch_embed usually expects (B, Channels, H, W). 
        # If your patch_embed treats T as 'Input Channels', then spatial dims are (C, N).
        data = data.permute(2, 0, 1) 

        # --- Global Crops ---
        # Global crops use the full extent
        global_crops = [data.clone(), data.clone()]
        
        # Metadata for global crops (All channels, All time patches)
        global_ch_idxs = torch.arange(self.C) # [0, 1, ..., 18]
        global_time_idxs = torch.arange(self.N) # [0, 1, ..., 29]
        
        # Pack metadata into a list matching global_crops list length
        global_indices = [
            {"ch": global_ch_idxs, "time": global_time_idxs},
            {"ch": global_ch_idxs.clone(), "time": global_time_idxs.clone()}
        ]

        # --- Local Crops ---
        local_crops = []
        local_indices = []
        
        for _ in range(self.n_local_crops):
            # A. Random Channels (Sorted)
            # Select indices, then sort them to preserve topological order
            ch_indices = torch.randperm(self.C)[:self.local_C].sort()[0]
            
            # B. Continuous Time Patches
            # Random start index such that [start, start + local_N] fits in N
            max_start = self.N - self.local_N
            start_t = torch.randint(0, max_start + 1, (1,)).item()
            time_indices = torch.arange(start_t, start_t + self.local_N)
            
            # C. Crop Data
            # data is (T, C, N). We select specific C and specific N.
            # Use index_select or slicing. Slicing is faster for contiguous time.
            # data[:, ch_indices, start_t : start_t + self.local_N]
            crop = data.index_select(1, ch_indices).index_select(2, time_indices)
            
            local_crops.append(crop)
            local_indices.append({"ch": ch_indices, "time": time_indices})

        return {
            "global_crops": global_crops,
            "global_indices": global_indices, # List of dicts
            "local_crops": local_crops,
            "local_indices": local_indices,   # List of dicts
        }
            
        # except Exception as e:
        #     print(f"Error loading {path}: {e}")
        #     # Return Dummy Data (omitted for brevity, assume similar structure)
        #     return {}