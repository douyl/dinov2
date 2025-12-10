# dinov2/models/vision_transformer.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from dinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block


logger = logging.getLogger("dinov2")


def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        # EEG Specific Arguments
        patch_time_dim=250,          # T: Number of time samples per patch
        num_channels=19,             # C: Total number of EEG channels available
        num_patches_per_channel=30,  # N: Total number of time patches per channel
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
    ):
        """
        Args:
            patch_time_dim (int): Input dimension T (time samples per patch). Default: 250.
            num_channels (int): Number of EEG channels (C). Default: 19.
            num_patches_per_channel (int): Number of patches per channel (N). Default: 30.
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1 # CLS token
        self.n_blocks = depth
        self.num_heads = num_heads

        self.patch_time_dim = patch_time_dim
        self.num_channels = num_channels
        self.num_patches_per_channel = num_patches_per_channel

        self.num_register_tokens = num_register_tokens  # Register token

        # Initialize PatchEmbed (Adapted for EEG)
        # in_chans corresponds to T (patch_time_dim) in this adaptation
        self.patch_embed = embed_layer(
            in_chans=patch_time_dim, 
            embed_dim=embed_dim, 
            num_channels=num_channels, 
            num_patches_per_channel=num_patches_per_channel
        )
        
        # We handle CLS token manually
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # --- EEG Positional Embeddings ---
        # Instead of a single fixed pos_embed, we use factorized embeddings for Channel and Time.
        # This allows flexible local cropping (e.g., selecting specific channels and time windows).
        
        # 1. Channel Embedding: Learnable vector for each of the EEG channels
        self.channel_embed = nn.Embedding(num_channels, embed_dim)
        
        # 2. Time Embedding: Learnable vector for each of the time patches
        self.time_embed = nn.Embedding(num_patches_per_channel, embed_dim)
        
        # Register tokens (optional extra CLS tokens)
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = np.linspace(0, drop_path_rate, depth).tolist()  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.channel_embed.weight, std=0.02)
        trunc_normal_(self.time_embed.weight, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def prepare_tokens_with_masks(self, x, masks=None, ch_idxs=None, time_idxs=None):
        """
        Args:
            x: Input tensor (B, T, C_crop, N_crop)
            masks: Boolean mask (B, C_crop*N_crop) where True indicates masked
            ch_idxs: Channel indices (B, C_crop)
            time_idxs: Time indices (B, N_crop)
        """
        
        # 1. Patch Embedding
        # (B, T, C, N) -> (B, C*N, D), where L = C*N
        x = self.patch_embed(x) 
        
        # 2. Masking
        if masks is not None:
            # masks shape: (B, L)
            # Broadcast mask_token to (B, L, D) and match dtype
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype), x)
        
        # 3. Add Positional Embeddings
        if ch_idxs is not None and time_idxs is not None:
            # (B, C_crop) -> (B, C_crop, D)
            ch_emb = self.channel_embed(ch_idxs)
            # (B, N_crop) -> (B, N_crop, D)
            t_emb = self.time_embed(time_idxs)
            
            # Combine: (B, C_crop, 1, D) + (B, 1, N_crop, D) -> (B, C_crop, N_crop, D)
            pos_embed_grid = ch_emb.unsqueeze(2) + t_emb.unsqueeze(1)
            
            # Flatten to match 'x' sequence order. 
            pos_embed = pos_embed_grid.flatten(1, 2)
            
            # Add pos_embed to x (which contains both visible patches and mask tokens)
            x = x + pos_embed
        else:
            raise ValueError("ch_idxs and time_idxs must be provided for positional embedding.")
        
        # 4. Append CLS Token
        B = x.shape[0]
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # 5. Append Register Tokens
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(B, -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward_features_list(self, x_list, masks_list, ch_idxs_list, time_idxs_list):
        # Process a list of inputs (e.g., global crops and local crops passed as a list)
        x = [
            self.prepare_tokens_with_masks(x, masks, ch_idxs, time_idxs) 
            for x, masks, ch_idxs, time_idxs in zip(x_list, masks_list, ch_idxs_list, time_idxs_list)
        ]
        
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None, ch_idxs=None, time_idxs=None):
        if isinstance(x, list):
            # If x is a list, we expect ch_idxs and time_idxs to be lists of matching length.
            # If they are not passed, we assume they are None (which might cause issues if pos embed is required).
            ch_idxs_list = ch_idxs if ch_idxs is not None else [None] * len(x)
            time_idxs_list = time_idxs if time_idxs is not None else [None] * len(x)
            masks_list = masks if masks is not None else [None] * len(x)
            return self.forward_features_list(x, masks_list, ch_idxs_list, time_idxs_list)

        x = self.prepare_tokens_with_masks(x, masks, ch_idxs, time_idxs)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        # Note: This method needs updating if used for inference with crops, 
        # passing default full-range indices if needed.
        # For now, we assume simple inference uses full frame.
        # Generating default indices for full frame:
        B = x.shape[0]
        device = x.device
        ch_idxs = torch.arange(self.num_channels, device=device).unsqueeze(0).expand(B, -1)
        time_idxs = torch.arange(self.num_patches_per_channel, device=device).unsqueeze(0).expand(B, -1)

        x = self.prepare_tokens_with_masks(x, ch_idxs=ch_idxs, time_idxs=time_idxs)
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        # Similar index generation as above
        B = x.shape[0]
        device = x.device
        ch_idxs = torch.arange(self.num_channels, device=device).unsqueeze(0).expand(B, -1)
        time_idxs = torch.arange(self.num_patches_per_channel, device=device).unsqueeze(0).expand(B, -1)

        x = self.prepare_tokens_with_masks(x, ch_idxs=ch_idxs, time_idxs=time_idxs)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        
        if reshape:
            # Reshape back to (B, C, N, D)
            B = x.shape[0]
            # Assumes full frame inference for reshaping logic
            C = self.num_channels
            # N might be inferred or fixed
            outputs = [
                out.reshape(B, C, -1, out.shape[-1]).contiguous() 
                for out in outputs
            ]
            
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_base(num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


# Other variants (small, large, giant) can be uncommented and adapted if needed
# For now, only vit_base is active as per user context.