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
        num_channels,       # Height (H): Number of EEG channels (e.g., 19)
        num_time_patches,   # Width (W): Number of time patches (e.g., 30)
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        # Map EEG dimensions to Image dimensions:
        # Height = Channels (Spatial dim), Width = Time (Temporal dim)
        self.height = num_channels 
        self.width = num_time_patches

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        # Aspect Ratio Settings:
        # We sample aspect ratio 'r' from a log-uniform distribution.
        # min_aspect = 0.15 (recommended for EEG) results in max_aspect = 1/0.15 ≈ 6.67.
        #
        # extreme case calculation (Assuming target_area ≈ 16 patches):
        # 1. Flat (Aspect ≈ 0.15): 
        #    h = sqrt(16 * 0.15) ≈ 1.5 -> 1 or 2 channels
        #    w = sqrt(16 / 0.15) ≈ 10.3 -> 10 time steps
        #    Physical Meaning: Simulates prolonged signal loss in a single channel.
        #
        # 2. Tall (Aspect ≈ 6.67): 
        #    h = sqrt(16 * 6.67) ≈ 10.3 -> 10 channels
        #    w = sqrt(16 / 6.67) ≈ 1.5 -> 1 or 2 time steps
        #    Physical Meaning: Simulates a transient artifact affecting half the brain.
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
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            # 1. Sample target area for this block
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            
            # 2. Sample aspect ratio (log-uniform)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            
            # 3. Calculate dimensions (H=Channels, W=Time)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            
            # 4. Check boundaries
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                
                # 5. Check overlap constraints
                # Ensure the new block contributes at least 1 new masked patch 
                # and doesn't exceed the quota.
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
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask