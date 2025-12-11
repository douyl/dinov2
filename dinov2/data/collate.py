# dinov2/data/collate.py

import torch
import random

def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
    # 1. Collate Crops (Images/Signals)
    # samples_list is a list，with length of Batch Size (B)
    # 每一项是一个字典，包含 'global_crops' (List of Tensors) 和 'local_crops' (List of Tensors)
    n_global_crops = len(samples_list[0]["global_crops"])  # for each batch
    n_local_crops = len(samples_list[0]["local_crops"])

    # List[ (250, 19, 30) ] -> Tensor (B * 2, 250, 19, 30)
    collated_global_crops = torch.stack([s["global_crops"][i] for i in range(n_global_crops) for s in samples_list])
    # List[ (250, C_local, N_local) ] -> Tensor (B * 8, 250, C_local, N_local)
    collated_local_crops = torch.stack([s["local_crops"][i] for i in range(n_local_crops) for s in samples_list])

    # 2. Collate Indices (Metadata for Positional Embedding)
    # Global indices: (B_global, C) and (B_global, N)
    # Since global crops are always full size, strictly speaking we could recreate them in model, 
    # but passing them keeps logic unified.
    global_ch_idxs = torch.stack([s["global_indices"][i]["ch"] for i in range(n_global_crops) for s in samples_list])
    global_time_idxs = torch.stack([s["global_indices"][i]["time"] for i in range(n_global_crops) for s in samples_list])
    
    # Local indices: (B_local, local_C) and (B_local, local_N)
    local_ch_idxs = torch.stack([s["local_indices"][i]["ch"] for i in range(n_local_crops) for s in samples_list])
    local_time_idxs = torch.stack([s["local_indices"][i]["time"] for i in range(n_local_crops) for s in samples_list])

    # 3. Masking Logic (Standard DINOv2 logic)
    """
        [Batch 级掩码调度逻辑：具体实例解析]
        这里决定了 Batch 内每张图具体遮挡多少比例。逻辑如下：
        1. 【确定数量】: 
        假设 B=6 (Total Crops), mask_prob=0.5。
        这意味着：有 3 张图需要遮挡 (Samples Masked)，剩下 3 张图完全不遮挡。
        2. 【制造难度阶梯】: 
        代码不希望大家难度一样，而是用 linspace 把 [0.1, 0.5] 的比例范围切分成 3 段：
        - 第 1 张图 (简单): 在 [0.10, 0.23] 之间随机选一个比例 (例如遮 80 个 Token)。
        - 第 2 张图 (中等): 在 [0.23, 0.36] 之间随机选一个比例 (例如遮 160 个 Token)。
        - 第 3 张图 (困难): 在 [0.36, 0.50] 之间随机选一个比例 (例如遮 240 个 Token)。
        3. 【混合与打乱】:
        - 生成完上述 3 个不同难度的 Mask 后，对其余 3 个样本生成全 0 Mask (不遮挡)。
        - 使用 shuffle 打乱列表顺序，防止模型根据 Batch 中的位置猜出难度。
        4. 【形状变换】:
        最后将 List 堆叠并 flatten(1)，把 (B, 19, 30) 压扁成 (B, 570) 喂给 Transformer。
    """
    B = len(collated_global_crops)  # actually B * n_global_crops
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        n_masked = int(N * random.uniform(prob_min, prob_max))
        masks_list.append(torch.BoolTensor(mask_generator(n_masked)))
        upperbound += int(N * prob_max)
        
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)
    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()
    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    return {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
        
        # New keys for Positional Embedding
        "global_ch_idxs": global_ch_idxs.long(),
        "global_time_idxs": global_time_idxs.long(),
        "local_ch_idxs": local_ch_idxs.long(),
        "local_time_idxs": local_time_idxs.long(),
    }