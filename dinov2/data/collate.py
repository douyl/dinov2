# dinov2/data/collate.py

import torch
import random

def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
    # samples_list 是一个 list，长度为 Batch Size (B)
    # 每一项是一个字典，包含 'global_crops' (List of Tensors) 和 'local_crops' (List of Tensors)
    n_global_crops = len(samples_list[0]["global_crops"])
    n_local_crops = len(samples_list[0]["local_crops"])

    # --- 1. 堆叠 Global Crops ---
    # 维度变换: List[ (19, 30, 250) ] -> Tensor (B * 2, 19, 30, 250)
    # [B_global, Channels, Patches, TimePoints]
    collated_global_crops = torch.stack([s["global_crops"][i] for i in range(n_global_crops) for s in samples_list])

    # --- 2. 堆叠 Local Crops ---
    # 维度变换: List[ (10, 30, 250) ] -> Tensor (B * 8, 10, 30, 250)
    # [B_local, Channels_subset, Patches, TimePoints]
    collated_local_crops = torch.stack([s["local_crops"][i] for i in range(n_local_crops) for s in samples_list])

    B = len(collated_global_crops) # 这里 B 实际上是 Batch_Size * n_global_crops
    
    # N 是 Patch/Token 的总数量。
    # 对于你的数据，N 应该等于 C * N_patches = 19 * 30 = 570。
    # 确保传入此函数的 n_tokens 参数是 570。
    N = n_tokens 
    
    # --- Mask 生成逻辑 ---
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        
        # mask_generator 在这里会被调用。
        # 它返回的 mask 形状通常是 (C, N_patches)，即 (19, 30)
        n_masked = int(N * random.uniform(prob_min, prob_max))
        
        # 生成 mask 并转为 BoolTensor
        # mask_generator(n_masked) -> (19, 30)
        masks_list.append(torch.BoolTensor(mask_generator(n_masked)))
        upperbound += int(N * prob_max)
        
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    # --- 3. 堆叠并 Flatten Masks ---
    # masks_list 是 B 个 (19, 30) 的 Tensor
    # torch.stack(masks_list) -> (B, 19, 30)
    # .flatten(1) -> (B, 570)
    # 最终 collated_masks 形状: (B, n_tokens)
    collated_masks = torch.stack(masks_list).flatten(1)
    
    # 计算非零索引，用于后续 gather 操作
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    return {
        # 输出给模型的 Global crops
        # 维度: (B_global, 19, 30, 250)
        "collated_global_crops": collated_global_crops.to(dtype),
        
        # 输出给模型的 Local crops
        # 维度: (B_local, 10, 30, 250)
        "collated_local_crops": collated_local_crops.to(dtype),
        
        # 输出给 Loss 计算的 Masks
        # 维度: (B_global, 570) -> 注意这里已经 Flatten 了空间维度
        "collated_masks": collated_masks,
        
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }