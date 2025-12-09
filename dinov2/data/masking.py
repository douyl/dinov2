# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import random
import math
import numpy as np


class MaskingGenerator:
    def __init__(
        self,
        num_channels,  # 新增参数：EEG通道数 (C)
        num_time_patches,  # 新增参数：每个通道的时间Patch数 (N)
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        # 将通道数视为 "高度"，时间Patch数视为 "宽度"
        # 这样可以利用原有的几何掩码逻辑来生成时间段掩码或通道掩码
        self.height = num_channels 
        self.width = num_time_patches

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        # 设置Mask块的长宽比范围
        # 小的 aspect (如 0.3) 会生成扁平的块（跨越时间长，通道少）
        # 大的 aspect 会生成瘦高的块（跨越通道多，时间短）
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(num_channels=%d, num_time_patches=%d -> [%d ~ %d], max = %r, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        # 返回形状 (C, N)
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            # 随机确定这次尝试生成的 Mask 块的目标面积（token数量）
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            
            # 随机确定 Mask 块的长宽比
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            
            # 计算 Mask 块的高度（涉及多少个通道）和宽度（涉及多少个时间步）
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            
            # 确保生成的块在数据边界内
            if w < self.width and h < self.height:
                # 随机选择起始的通道索引和时间索引
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                
                # 检查与已有的 Mask 是否有过多重叠
                # 如果这是一个有效的 Mask 区域（未被完全覆盖且不超出最大限制）
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        # 初始化全 0 的 Mask 矩阵，形状为 (C, N)
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        
        # 循环直到达到目标 Mask 数量
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask