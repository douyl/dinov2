# dinov2/data/datasets/eeg_dataset.py

import torch
import numpy as np
import os
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    def __init__(self, root, C=19, N=30, T=250):
        super().__init__()
        self.root = root
        self.C = C
        self.N = N
        self.T = T
        # 获取目录下所有npy文件
        self.files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.npy')]
        
        # 简单的检查一下文件数量
        if len(self.files) == 0:
            print(f"警告: 在 {root} 没有找到 .npy 文件")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # 读取 .npy 文件
        path = self.files[index]
        try:
            # 假设数据格式是 (C, N, T) -> (19, 30, 250)
            data_np = np.load(path)
            
            # 转为 Tensor
            data = torch.from_numpy(data_np).float()
            
            # DINOv2的Transformer输入通常是一序列的Token。
            # 这里我们需要把 (C, N, T) 变成 (Num_Tokens, Embed_Dim)。
            # Num_Tokens = C * N, Embed_Dim = T
            # 变换维度: (C, N, T) -> (C * N, T)
            data_flattened = data.view(-1, self.T) # Shape: (570, 250)
            
            # DINOv2 训练代码期望返回一个字典，包含 'global_crops' 和 'local_crops'。
            # 对于EEG，我们暂时生成2个相同的 Global Crops 用于计算 Loss (类似 SimCLR/DINO 的双视角)。
            # 如果你有数据增强策略（如随机Mask通道，随机截取时间段），应该在这里应用。
            # 目前为了跑通代码，我们直接复制。
            
            return {
                "global_crops": [data_flattened.clone(), data_flattened.clone()],
                "local_crops": [data_flattened.clone(), data_flattened.clone()], # 暂时不需要 local crops，可以留空
            }
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # 返回一个全0的dummy数据防止崩坏
            dummy = torch.zeros(self.C * self.N, self.T)
            return {
                "global_crops": [dummy, dummy],
                "local_crops": []
            }