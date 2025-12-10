# dinov2/data/datasets/eeg_dataset.py

import torch
import numpy as np
import os
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    def __init__(self, data_root, num_channels=19, num_patches_per_channel=30, patch_time_dim=250, local_crop_channels=10):
        super().__init__()
        self.data_root = data_root
        self.C = num_channels  # 19
        self.N = num_patches_per_channel # 30
        self.T = patch_time_dim # 250
        self.local_crop_channels = local_crop_channels
        
        # 预先筛选 .npy 文件
        self.files = []
        if os.path.exists(data_root):
             self.files = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith('.npy')]
        
        if len(self.files) == 0:
            print(f"警告: 在 {data_root} 没有找到 .npy 文件")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        try:
            # 1. 加载数据
            # 原始 numpy 形状: (C, N, T) -> (19, 30, 250)
            data_np = np.load(path)
            
            # 2. 转为 Tensor (float)
            data = torch.from_numpy(data_np).float()
            
            # ============================
            # [新增] 方案 A: Per-Sample Z-Score 归一化
            # ============================
            # 对当前样本的所有数据计算均值和方差
            # 目的：消除个体差异和采集设备差异，让所有样本分布在 0 附近
            mean = data.mean()
            std = data.std()
            # 加上 1e-6 防止除以 0（但是太小了！！！）
            # data = (data - mean) / (std + 1e-6)
            data = (data - mean) / std
            
            # ============================
            # [新增] 维度调整: 把 T (特征维度) 放到最前面
            # ============================
            # 原始: (C, N, T) -> (0, 1, 2)
            # 目标: (T, C, N) -> (2, 0, 1) => (250, 19, 30)
            data = data.permute(2, 0, 1)

            # --- Global Crops ---
            # 现在的 shape 是 (250, 19, 30)
            global_crops = [data.clone(), data.clone()]

            # --- Local Crops ---
            # 随机选择 10 个通道
            # 注意：现在的维度是 (T, C, N)，C 在第 1 维
            local_crops = []
            for _ in range(8):
                # 随机生成通道索引 (在 0 到 18 之间选 10 个)
                indices = torch.randperm(self.C)[:self.local_crop_channels]
                
                # 选取通道: data[:, indices, :]
                # 结果形状: (250, 10, 30)
                crop = data[:, indices, :].clone()
                local_crops.append(crop)

            return {
                "global_crops": global_crops, # List of (250, 19, 30)
                "local_crops": local_crops,   # List of (250, 10, 30)
            }
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # 返回 Dummy 数据保持维度一致 (T, C, N)
            dummy_global = torch.zeros(self.T, self.C, self.N)
            dummy_local = torch.zeros(self.T, self.local_crop_channels, self.N)
            return {
                "global_crops": [dummy_global, dummy_global],
                "local_crops": [dummy_local] * 8 
            }