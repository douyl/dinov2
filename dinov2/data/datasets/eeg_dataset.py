# dinov2/data/datasets/eeg_dataset.py

import torch
import numpy as np
import os
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    def __init__(self, data_root, num_channels=19, num_patches_per_channel=30, patch_time_dim=250, local_crop_channels=10):
        super().__init__()
        self.data_root = data_root
        self.C = num_channels
        self.N = num_patches_per_channel
        self.T = patch_time_dim
        self.local_crop_channels = local_crop_channels
        self.files = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith('.npy')]
        
        if len(self.files) == 0:
            print(f"警告: 在 {data_root} 没有找到 .npy 文件")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        try:
            # 加载数据 (C, N, T) -> (19, 30, 250)
            data_np = np.load(path)
            data = torch.from_numpy(data_np).float()
            
            # --- Global Crops ---
            # 保持原始维度 (C, N, T)，不要在这里 Flatten
            # 如果有数据增强（比如时间轴的随机Mask或Jitter），在这里对 global_1 和 global_2 分别做
            global_crops = [data.clone(), data.clone()]

            # --- Local Crops ---
            # 随机选择 10 个通道
            # 维度变化: (19, 30, 250) -> (10, 30, 250)
            local_crops = []
            # 假设我们需要 8 个 local crops (DINOv2 默认配置)
            for _ in range(8):
                # 随机生成通道索引
                indices = torch.randperm(self.C)[:self.local_crop_channels]
                # 选取通道
                crop = data[indices].clone()
                local_crops.append(crop)

            return {
                "global_crops": global_crops, # List of (19, 30, 250)
                "local_crops": local_crops,   # List of (10, 30, 250)
            }
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # 返回 Dummy 数据保持维度一致
            dummy_global = torch.zeros(self.C, self.N, self.T)
            dummy_local = torch.zeros(self.local_crop_channels, self.N, self.T)
            # 生成对应数量的 dummy crops
            return {
                "global_crops": [dummy_global, dummy_global],
                "local_crops": [dummy_local] * 8 
            }