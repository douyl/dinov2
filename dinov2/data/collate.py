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
    B = len(collated_global_crops)
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