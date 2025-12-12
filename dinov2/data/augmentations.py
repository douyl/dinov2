# dinov2/data/augmentations.py

import logging
import torch
import random
import numpy as np

logger = logging.getLogger("dinov2")

class DataAugmentationDINO(object):
    def __init__(
        self,
        # Dimensions
        global_crop_size_channels=19,
        global_crop_size_patches=30,
        patch_time_dim=250,      
        local_crops_number=8,
        local_crop_size_channels=10,
        local_crop_size_patches=15,
        
        # Augmentation Probabilities
        global_aug_probs=(1.0, 0.1), 
        local_aug_prob=0.5,          
        
        # Augmentation Hyperparameters
        scale_prob=0.5,          
        scale_sigma=0.2,         
        noise_prob=0.5,          
        noise_std_range=(0.001, 0.01),   # (0.05, 0.1)
        dropout_ch_prob=0.05,       
        dropout_max_channels=3,     
        dropout_as_noise_prob=0.5,  # [NEW] Probability to replace channel with noise instead of zeros
        time_shift_prob=0.3,     
        max_time_shift_ratio=0.05, 
        phase_perturb_prob=0.2,  
        phase_perturb_rad=0.2,
    ):
        self.C = global_crop_size_channels
        self.N = global_crop_size_patches
        self.T = patch_time_dim
        self.total_time = self.N * self.T
        
        self.n_local_crops = local_crops_number
        self.local_C = local_crop_size_channels
        self.local_N = local_crop_size_patches
        
        # Probabilities
        self.global_aug_probs = global_aug_probs
        self.local_aug_prob = local_aug_prob
        
        # Hyperparams
        self.scale_prob = scale_prob
        self.scale_sigma = scale_sigma
        self.noise_prob = noise_prob
        self.noise_std_range = noise_std_range
        self.dropout_ch_prob = dropout_ch_prob
        self.dropout_max_channels = dropout_max_channels
        self.dropout_as_noise_prob = dropout_as_noise_prob
        self.time_shift_prob = time_shift_prob
        self.max_time_shift = int(self.total_time * max_time_shift_ratio)
        self.phase_perturb_prob = phase_perturb_prob
        self.phase_perturb_rad = phase_perturb_rad

        logger.info(f"EEG Augmentations Initialized. Global Probs: {global_aug_probs}, Local Prob: {local_aug_prob}")

    def _apply_physics_augmentations(self, x_flat):
        """
        Apply augmentations on continuous signal (C, Total_Time).
        """
        C, Total_T = x_flat.shape
        device = x_flat.device
        
        # 1. Time Shift (Global Roll) - preserves synchronization
        if self.time_shift_prob > 0 and random.random() < self.time_shift_prob:
            shift = random.randint(-self.max_time_shift, self.max_time_shift)
            if shift != 0:
                x_flat = torch.roll(x_flat, shifts=shift, dims=1)

        # 2. Phase Perturbation (Spectral Domain) - Shared across channels
        if self.phase_perturb_prob > 0 and random.random() < self.phase_perturb_prob:
            fft_x = torch.fft.rfft(x_flat, dim=1) # (C, Freq)
            Freq_dim = fft_x.shape[1]
            
            # [MODIFIED] Generate SHARED phase noise for all channels to preserve relative phase
            # Shape: (1, Freq)
            phase_noise = (torch.rand(1, Freq_dim, device=device) * 2 - 1) * self.phase_perturb_rad
            
            # Apply same rotation to all channels
            rotation = torch.complex(torch.cos(phase_noise), torch.sin(phase_noise))
            x_flat = torch.fft.irfft(fft_x * rotation, n=Total_T, dim=1)

        # 3. Amplitude Scale (Per Channel Independent)
        if self.scale_prob > 0 and random.random() < self.scale_prob:
            # shape: (C, 1)
            scales = torch.exp(torch.randn(C, 1, device=device) * self.scale_sigma)
            x_flat = x_flat * scales

        # 4. Gaussian Noise (Per Channel Independent STD)
        if self.noise_prob > 0 and random.random() < self.noise_prob:
            min_std, max_std = self.noise_std_range
            # [MODIFIED] Generate independent std for each channel: (C, 1)
            stds = torch.rand(C, 1, device=device) * (max_std - min_std) + min_std
            noise = torch.randn_like(x_flat) * stds
            x_flat = x_flat + noise

        # 5. Channel Dropout / Corruption
        if self.dropout_ch_prob > 0 and random.random() < self.dropout_ch_prob:
            num_drop = random.randint(1, self.dropout_max_channels)
            drop_indices = torch.randperm(C, device=device)[:num_drop]
            
            x_flat = x_flat.clone()
            
            # [MODIFIED] Randomly choose between Zero-masking or Noise-masking
            if random.random() < self.dropout_as_noise_prob:
                # Replace with pure noise (simulate high impedance/artifact)
                # Usually artifacts have high amplitude, let's use 2x max std
                artifact_std = self.noise_std_range[1] * 5.0 
                noise = torch.randn(num_drop, Total_T, device=device) * artifact_std
                x_flat[drop_indices, :] = noise
            else:
                # Replace with zeros (simulate disconnected)
                x_flat[drop_indices, :] = 0.0
            
        return x_flat

    def _augment_wrapper(self, data, p_apply):
        """
        Input: (T, C, N) -> (250, 19, 30)
        Logic: Permute -> (C, N, T) -> Flatten -> (C, N*T) -> Augment -> Reshape Back
        """
        if random.random() >= p_apply:
            return data

        # (T, C, N) -> (C, N, T)
        x_cont = data.permute(1, 2, 0)
        C_curr, N_curr, T_curr = x_cont.shape
        
        # (C, N, T) -> (C, N*T)
        x_cont = x_cont.reshape(C_curr, -1)

        # Apply Physics Augs
        x_aug_cont = self._apply_physics_augmentations(x_cont)

        # Reshape back: (C, N*T) -> (C, N, T) -> (T, C, N)
        x_aug = x_aug_cont.reshape(C_curr, N_curr, T_curr).permute(2, 0, 1)
        
        return x_aug

    def __call__(self, data):
        # Assertions
        assert data.ndim == 3, f"Input must be (T, C, N), got {data.shape}"
        assert data.shape == (self.T, self.C, self.N), f"Shape mismatch: {data.shape} vs {(self.T, self.C, self.N)}"

        output = {}

        # --- Global Crops ---
        global_crops = []
        global_indices = []
        full_ch_idxs = torch.arange(self.C)
        full_time_idxs = torch.arange(self.N)

        for i in range(2):
            # Check range of probs list
            p = self.global_aug_probs[i] if i < len(self.global_aug_probs) else 1.0
            
            crop = self._augment_wrapper(data, p_apply=p)
            global_crops.append(crop)
            global_indices.append({"ch": full_ch_idxs.clone(), "time": full_time_idxs.clone()})

        output["global_crops"] = global_crops
        output["global_indices"] = global_indices

        # --- Local Crops ---
        local_crops = []
        local_indices = []

        for _ in range(self.n_local_crops):
            # 1. Slicing (Geometric Crop)
            # A. Channels (Sorted)
            crop_ch_idxs = torch.randperm(self.C)[:self.local_C].sort()[0]
            # B. Time (Continuous)
            max_start = self.N - self.local_N
            start_t = torch.randint(0, max_start + 1, (1,)).item()
            crop_time_indices = torch.arange(start_t, start_t + self.local_N)
            
            # Crop Data: (T, C, N) -> (T, local_C, local_N)
            crop_data = data.index_select(1, crop_ch_idxs).index_select(2, crop_time_indices)
            
            # 2. Augmentation (on the crop)
            crop_data_aug = self._augment_wrapper(crop_data, p_apply=self.local_aug_prob)
            
            local_crops.append(crop_data_aug)
            local_indices.append({"ch": crop_ch_idxs, "time": crop_time_indices})

        output["local_crops"] = local_crops
        output["local_indices"] = local_indices
        # output["offsets"] = ()

        return output