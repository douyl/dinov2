import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import torch
import numpy as np
from functools import partial
from torch.utils.data import DataLoader

# 引入你的模块
# 假设你已经把 EEGDataset 放到了 dinov2/data/datasets/eeg_dataset.py
from dinov2.data.datasets.eeg_dataset import EEGDataset
# 引入 MaskingGenerator 和 collate
from dinov2.data.masking import MaskingGenerator
from dinov2.data.collate import collate_data_and_cast

# 为了让 collate 正常工作，我们需要模拟一个 Config 对象或直接传参
# 这里我们直接定义需要的参数
class MockConfig:
    def __init__(self):
        # Masking 相关的参数
        self.mask_ratio_min_max = (0.1, 0.5)  # mask 比例范围
        self.mask_sample_probability = 0.5    # 样本被 mask 的概率
        self.batch_size = 2

def main():
    print("=== 开始测试 EEG DINOv2 Data Pipeline ===")

    # 1. 基础参数设置
    # 你的数据维度
    C = 19   # 通道数
    N = 30   # 时间Patch数
    T = 250  # 每个Patch的时间点数 (Embedding Dim)
    
    n_tokens = C * N  # 总 Token 数 = 570
    
    print(f"参数设置: C={C}, N={N}, T={T}, Total Tokens={n_tokens}")

    # 2. 检查数据文件是否存在，如果不存在创建一个 Dummy 用于测试
    data_path = "samples"
    sample_file = os.path.join(data_path, "aaaaaaaa_s001_t000.npy")
    
    if not os.path.exists(sample_file):
        print(f"[提示] 未找到 {sample_file}，正在创建 Dummy 数据...")
        os.makedirs(data_path, exist_ok=True)
        # 生成 (19, 30, 250) 的随机数据
        dummy_data = np.random.randn(C, N, T).astype(np.float32)
        np.save(sample_file, dummy_data)
    else:
        print(f"[提示] 发现数据文件: {sample_file}")

    # 3. 实例化 MaskingGenerator
    # 注意：这里使用你修改后的 __init__ 参数 (num_channels, num_time_patches)
    print("正在初始化 MaskingGenerator...")
    mask_generator = MaskingGenerator(
        num_channels=C,
        num_time_patches=N,
        max_num_patches=0.5 * C * N, # 最多 Mask 掉一半
    )
    print("MaskingGenerator 初始化成功:", mask_generator)

    # 4. 准备 Collate Function
    # DINOv2 使用 partial 来固定 collate 的部分参数
    cfg = MockConfig()
    inputs_dtype = torch.float32 # EEG数据一般是 float32

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.mask_ratio_min_max,
        mask_probability=cfg.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    # 5. 实例化 Dataset
    print("正在加载 EEGDataset...")
    dataset = EEGDataset(root=data_path, C=C, N=N, T=T)
    print(f"Dataset 长度: {len(dataset)}")

    # 6. 实例化 DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=0, # 本地测试建议设为 0
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False # 测试时为了保证能取到数据，不 drop last
    )

    # 7. 测试迭代
    print("\n>>> 正在尝试从 DataLoader 读取一个 Batch...")
    try:
        for i, batch in enumerate(data_loader):
            print("\n[成功] 读取到 Batch 数据！包含以下 Keys:")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"  - {k}: shape={v.shape}, dtype={v.dtype}")
                else:
                    print(f"  - {k}: {v}")

            # 验证一些关键形状
            # collated_global_crops 应该是 (B * 2, n_tokens, embed_dim) -> (2, 570, 250)
            global_crops = batch["collated_global_crops"]
            masks = batch["collated_masks"]
            
            expected_crops_shape = (cfg.batch_size * 2, n_tokens, T)
            # collated_masks 是 flatten 后的，形状通常是 (B * 2, n_tokens) 或者 (B*2*n_tokens) 取決於實現，
            # 原版 collate.py 里是: collated_masks = torch.stack(masks_list).flatten(1)
            # 所以形状应该是 (B_total, n_tokens)
            expected_mask_shape = (cfg.batch_size * 2, n_tokens)

            assert global_crops.shape == expected_crops_shape, \
                f"Global Crops 形状错误: 期望 {expected_crops_shape}, 实际 {global_crops.shape}"
            
            assert masks.shape == expected_mask_shape, \
                f"Masks 形状错误: 期望 {expected_mask_shape}, 实际 {masks.shape}"

            print("\n[通过] 形状检查通过。")
            
            # 简单展示一下 Mask 的情况
            mask_ratio = masks.float().mean().item()
            print(f"当前 Batch 的实际 Mask 比例: {mask_ratio:.4f}")
            
            # 只测试一个 Batch 就退出
            break
            
    except Exception as e:
        print(f"\n[错误] DataLoader 迭代失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()