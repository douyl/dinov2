# dinov2/data/collate.py
# 保留源代码，基本可以直接复用。
# 关键在于调用此函数时传入的 samples_list 里的数据维度，以及 n_tokens 的计算。

import torch
import random

def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
    # 获取 global crops 的数量 (通常是2)
    n_global_crops = len(samples_list[0]["global_crops"])
    # 获取 local crops 的数量 (通常是8)
    n_local_crops = len(samples_list[0]["local_crops"])

    # 堆叠 Global Crops. 
    # samples_list[i]["global_crops"][j] 的形状应该是 (Num_Tokens, Embed_Dim) 即 (570, 250)
    collated_global_crops = torch.stack([s["global_crops"][i] for i in range(n_global_crops) for s in samples_list])

    # 处理 Local Crops (如果有)
    collated_local_crops = torch.stack([s["local_crops"][i] for i in range(n_local_crops) for s in samples_list])

    B = len(collated_global_crops)
    N = n_tokens # 这里 N 是 token 的数量 (570)
    
    # 下面是生成 Mask 的逻辑
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        
        # 这里的 mask_generator 必须能接受一个整数 (mask的数量) 并返回一个 BoolTensor。
        # 只要我们在外部定义好适合 EEG 数据的 mask_generator，这里就不需要改。
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
    }