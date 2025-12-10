# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    EEG to patch embedding: (B, C, N, T) -> (B, C*N, D)

    Args:
        in_chans: Input dimension T (time samples per patch). Default: 250.
        embed_dim: Embedding dimension. Default: 768.
        norm_layer: Normalization layer.
        flatten_embedding: If True, flatten the input to (B, C*N, D).
        num_channels: Number of EEG channels (C), used for calculating num_patches. Default: 19.
        num_patches_per_channel: Number of patches per channel (N). Default: 30.
    """

    def __init__(
        self,
        in_chans: int = 250,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
        num_channels: int = 19,
        num_patches_per_channel: int = 30,
    ) -> None:
        super().__init__()

        # In this EEG context, in_chans corresponds to the time dimension T
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding

        # Calculate total number of patches (C * N) to maintain consistency with ViT logic
        self.num_patches = num_channels * num_patches_per_channel

        # Use Linear layer for projection: T -> embed_dim
        self.proj = nn.Linear(in_chans, embed_dim)
        
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, N, T)
        """
        B, C, N, T = x.shape
        
        # Verify input dimension T matches in_chans
        assert T == self.in_chans, f"Input time dimension {T} does not match in_chans {self.in_chans}"
        assert C * N == self.num_patches, f"Input patches {C*N} do not match expected num_patches {self.num_patches}"
        
        # Flatten Channels (C) and Patches (N) dimensions: (B, C, N, T) -> (B, C*N, T)
        x = x.flatten(1, 2)

        # Project to embedding dimension: (B, C*N, T) -> (B, C*N, embed_dim)
        x = self.proj(x)

        # Apply normalization
        x = self.norm(x)

        # Reshape back to (B, C, N, embed_dim) if not flattening
        if not self.flatten_embedding:
            x = x.reshape(B, C, N, self.embed_dim)

        return x