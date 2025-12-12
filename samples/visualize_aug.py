import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import mne
import random

# ==========================================
# 1. 粘贴或导入 DataAugmentationEEG 类
# ==========================================
# (为了方便直接运行，这里完整粘贴了上一版确定的 DataAugmentationEEG 类)
class DataAugmentationEEG(object):
    def __init__(
        self,
        global_crop_size_channels=19,
        global_crop_size_patches=30,
        patch_time_dim=250,      
        local_crops_number=8,
        local_crop_size_channels=10,
        local_crop_size_patches=15,
        global_aug_probs=(1.0, 0.1), 
        local_aug_prob=0.5,          
        scale_prob=0.5,          
        scale_sigma=0.2,         
        noise_prob=0.5,          
        noise_std_range=(0.001, 0.01), 
        mask_ch_prob=0.05,       
        mask_max_channels=3,     
        mask_as_noise_prob=0.5,
        time_shift_prob=0.3,     
        max_time_shift_ratio=0.05, 
        phase_perturb_prob=0.0,  
        phase_perturb_rad=0.2,
    ):
        self.C = global_crop_size_channels
        self.N = global_crop_size_patches
        self.T = patch_time_dim
        self.total_time = self.N * self.T
        
        self.n_local_crops = local_crops_number
        self.local_C = local_crop_size_channels
        self.local_N = local_crop_size_patches
        
        self.global_aug_probs = global_aug_probs
        self.local_aug_prob = local_aug_prob
        
        self.scale_prob = scale_prob
        self.scale_sigma = scale_sigma
        self.noise_prob = noise_prob
        self.noise_std_range = noise_std_range
        self.mask_ch_prob = mask_ch_prob
        self.mask_max_channels = mask_max_channels
        self.mask_as_noise_prob = mask_as_noise_prob
        self.time_shift_prob = time_shift_prob
        self.max_time_shift = int(self.total_time * max_time_shift_ratio)
        self.phase_perturb_prob = phase_perturb_prob
        self.phase_perturb_rad = phase_perturb_rad

    def _apply_physics_augmentations(self, x_flat):
        C, Total_T = x_flat.shape
        device = x_flat.device
        
        # 1. Time Shift
        if self.time_shift_prob > 0 and random.random() < self.time_shift_prob:
            shift = random.randint(-self.max_time_shift, self.max_time_shift)
            if shift != 0:
                x_flat = torch.roll(x_flat, shifts=shift, dims=1)

        # 2. Phase Perturbation
        if self.phase_perturb_prob > 0 and random.random() < self.phase_perturb_prob:
            fft_x = torch.fft.rfft(x_flat, dim=1) 
            Freq_dim = fft_x.shape[1]
            phase_noise = (torch.rand(1, Freq_dim, device=device) * 2 - 1) * self.phase_perturb_rad
            rotation = torch.complex(torch.cos(phase_noise), torch.sin(phase_noise))
            x_flat = torch.fft.irfft(fft_x * rotation, n=Total_T, dim=1)

        # 3. Amplitude Scale
        if self.scale_prob > 0 and random.random() < self.scale_prob:
            scales = torch.exp(torch.randn(C, 1, device=device) * self.scale_sigma)
            x_flat = x_flat * scales

        # 4. Gaussian Noise
        if self.noise_prob > 0 and random.random() < self.noise_prob:
            min_std, max_std = self.noise_std_range
            stds = torch.rand(C, 1, device=device) * (max_std - min_std) + min_std
            noise = torch.randn_like(x_flat) * stds
            x_flat = x_flat + noise

        # 5. Channel Dropout
        if self.mask_ch_prob > 0 and random.random() < self.mask_ch_prob:
            num_drop = random.randint(1, self.mask_max_channels)
            drop_indices = torch.randperm(C, device=device)[:num_drop]
            x_flat = x_flat.clone()
            if random.random() < self.mask_as_noise_prob:
                artifact_std = self.noise_std_range[1] * 5.0 
                noise = torch.randn(num_drop, Total_T, device=device) * artifact_std
                x_flat[drop_indices, :] = noise
            else:
                x_flat[drop_indices, :] = 0.0
            
        return x_flat

    def _augment_wrapper(self, data, p_apply):
        if random.random() >= p_apply:
            return data
        x_cont = data.permute(1, 2, 0) # (T,C,N) -> (C,N,T)
        C_curr, N_curr, T_curr = x_cont.shape
        x_cont = x_cont.reshape(C_curr, -1) # -> (C, N*T)
        x_aug_cont = self._apply_physics_augmentations(x_cont)
        x_aug = x_aug_cont.reshape(C_curr, N_curr, T_curr).permute(2, 0, 1) # -> (T,C,N)
        return x_aug

    def __call__(self, data):
        output = {}
        global_crops = []
        for i in range(2):
            p = self.global_aug_probs[i] if i < len(self.global_aug_probs) else 1.0
            crop = self._augment_wrapper(data, p_apply=p)
            global_crops.append(crop)
        output["global_crops"] = global_crops
        
        # Local crops (省略以简化可视化)
        # ... 实际使用时这里会有 Local crops 逻辑 ...
        return output

# ==========================================
# 2. 数据加载与预处理逻辑
# ==========================================

def _normalize_per_channel(x):
    """
    User provided normalization logic.
    x shape: (C, N, T)
    """
    # Calculate mean/std along (N, T) dimensions
    mean = x.mean(axis=(1, 2), keepdims=True)
    std = x.std(axis=(1, 2), keepdims=True)
    # Avoid division by zero
    return np.divide(x - mean, std, out=np.zeros_like(x), where=std > 1e-8)

def load_edf_and_process(file_path):
    """
    1. Read EDF
    2. Crop 60s - 90s
    3. Resample to 250Hz
    4. Reshape to (C, N, T)
    5. Normalize
    6. Permute to (T, C, N)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"Loading {file_path}...")
    # 1. Read Raw
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    
    # 2. Crop (tmin=60, tmax=90)
    # 注意：raw.crop 是 in-place 操作
    print("Cropping data (60s - 90s)...")
    raw.crop(tmin=60, tmax=90, include_tmax=False)
    
    # 3. Resample & Select Channels
    target_fs = 250
    if int(raw.info['sfreq']) != target_fs:
        print(f"Resampling to {target_fs}Hz...")
        raw.resample(target_fs)
    
    # 获取 numpy 数据 (C, Total_Points)
    # 乘 1e6 把 V 转为 uV，这是 EEG 常用单位，防止数值过小
    data_raw = raw.get_data() * 1e6 
    
    # 截取前 19 个通道
    target_channels = 19
    if data_raw.shape[0] > target_channels:
        data_raw = data_raw[:target_channels, :]
    elif data_raw.shape[0] < target_channels:
        # Pad with zeros if channels are missing (safety check)
        pad = np.zeros((target_channels - data_raw.shape[0], data_raw.shape[1]))
        data_raw = np.vstack([data_raw, pad])

    # 检查时间点数
    # 30s * 250Hz = 7500 points
    target_points = 30 * target_fs
    if data_raw.shape[1] != target_points:
        print(f"Warning: Data length is {data_raw.shape[1]}, expected {target_points}. Truncating/Padding.")
        if data_raw.shape[1] > target_points:
             data_raw = data_raw[:, :target_points]
        else:
             pad = np.zeros((target_channels, target_points - data_raw.shape[1]))
             data_raw = np.hstack([data_raw, pad])

    # 4. Reshape to (C, N, T)
    # C=19, N=30, T=250 -> Total=7500 matches
    C = target_channels
    N = 30
    T = 250
    data_reshaped = data_raw.reshape(C, N, T)
    
    # 5. Normalize (Z-score)
    print("Applying Z-score normalization...")
    data_norm = _normalize_per_channel(data_reshaped)
    
    # 6. Permute to (T, C, N) for Model Input
    # (C, N, T) -> (T, C, N)
    data_tensor = torch.from_numpy(data_norm).float()
    data_tensor = data_tensor.permute(2, 0, 1) # 2->0(T), 0->1(C), 1->2(N)
    
    print(f"Processed Data Shape: {data_tensor.shape} (T, C, N)")
    return data_tensor

# ==========================================
# 3. 绘图与主函数
# ==========================================

def plot_comparison(original, augmented, title, fs=250):
    """
    Flatten input from (T, C, N) back to (C, Time) for plotting.
    """
    def to_flat(x):
        # (T, C, N) -> (C, N, T) -> (C, N*T)
        return x.permute(1, 2, 0).reshape(x.shape[1], -1).cpu().numpy()

    org_flat = to_flat(original)
    aug_flat = to_flat(augmented)
    
    channels_to_plot = [0, 5, 8, 10, 15, 18] # Plot specific channels
    total_time_points = org_flat.shape[1]
    time_axis = np.arange(total_time_points) / fs

    fig, axes = plt.subplots(len(channels_to_plot), 1, figsize=(12, 10), sharex=True)
    if len(channels_to_plot) == 1: axes = [axes]
    
    for i, ch_idx in enumerate(channels_to_plot):
        if ch_idx >= org_flat.shape[0]: continue
        ax = axes[i]
        
        # Original
        ax.plot(time_axis, org_flat[ch_idx], label='Original', color='black', alpha=0.6, linewidth=0.8)
        # Augmented
        ax.plot(time_axis, aug_flat[ch_idx], label='Augmented', color='red', alpha=0.7, linewidth=0.8, linestyle='--')
        
        ax.set_ylabel(f'Ch {ch_idx}')
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, alpha=0.2)
    
    plt.xlabel('Time (s)')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # --- 文件路径 ---
    FILE_PATH = "/media/douyl/Disk4T/douyl/IDEA_Lab/Project_BCI/dinov2/samples/aaaaaaaa_s001_t000.edf"
    
    # --- 增强参数配置 (在这里手动修改以测试不同效果) ---
    aug_params = dict(
        # 1. 信号缩放
        scale_prob=0,           
        scale_sigma=0.2,          
        
        # 2. 噪声
        noise_prob=0,           
        noise_std_range=(0.05, 0.1),
        
        # 3. 通道 Mask
        mask_ch_prob=0,         
        mask_max_channels=3,
        mask_as_noise_prob=0.5,
        
        # 4. 时间位移
        time_shift_prob=0,      
        max_time_shift_ratio=0.05,
        
        # 5. 相位扰动
        phase_perturb_prob=0,   
        phase_perturb_rad=0.2,    
    )

    try:
        # 1. 加载并预处理数据
        data_tensor = load_edf_and_process(FILE_PATH)
        
        # 2. 初始化增强模块
        augmenter = DataAugmentationEEG(
            global_crop_size_channels=19,
            global_crop_size_patches=30,
            patch_time_dim=250,
            global_aug_probs=[1.0, 1.0], # 强制让两个 global crops 都做增强方便观察
            **aug_params
        )

        # 3. 执行增强
        print("Applying Augmentation...")
        output = augmenter(data_tensor)
        
        # 取出第一个增强后的 Global Crop
        aug_tensor = output["global_crops"][0]

        # 4. 画图对比
        print("Plotting comparison...")
        plot_comparison(
            data_tensor, 
            aug_tensor, 
            title=f"EEG Augmentation Check\nParams: {aug_params}"
        )
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()