"""
Custom implementation of the Decoder Transformer Block with added MTP Head Extension at the final layer.
"""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Union, Optional, Tuple
import einops
import math
from jaxtyping import Float, Int 
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCacheEntry
from transformer_lens.utils import get_offset_position_ids
from transformer_lens.utilities.addmm import batch_addmm
from transformer_lens.hook_points import HookPoint

from model.config import GPT2MTPConfig
from model.utils_mtp import (
    simple_attn_linear,
    calculate_attn_scores, 
    apply_causal_mask,
)

class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""
    
    def __init__(self, config: GPT2MTPConfig, length: Optional[int] = None):
        super().__init__()
        self.cfg = config
        if length is None: 
            self.length = self.cfg.d_model
        else:
            self.length = length
        self.weight = nn.Parameter(torch.ones(self.length, dtype=self.cfg.dtype))
        self.bias = nn.Parameter(torch.zeros(self.length, dtype=self.cfg.dtype)) 
        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos, 1]
        # Hook_normalized is on the LN output
        self.hook_normalized = HookPoint()  # [batch, pos, length]
    
    def setup_hooks(self, parent_name: str):
        self.hook_scale.name = f"{parent_name}.hook_scale"
        self.hook_normalized.name = f"{parent_name}.hook_normalized"

    def forward(self, x: Float[torch.Tensor, "batch pos d_model"]) -> Float[torch.Tensor, "batch pos d_model"]:
        x = x - x.mean(-1, keepdim=True)  # [batch, pos, length]
        scale = self.hook_scale(x.pow(2).mean(-1, keepdim=True) + 1e-5).sqrt()
        x = x / scale  # [batch, pos, length]
        return self.hook_normalized(x * self.weight + self.bias).to(self.cfg.dtype)

# MLP class from transformer_lens.components.mlps.mlp 
class MLP(nn.Module):

    def __init__(self, config: GPT2MTPConfig):
        super().__init__()
        self.cfg = config
        self.hooks_enabled = False  # default off
        
        if self.cfg.act_function == "gelu":
            self.act = nn.GELU()

        self.ln = LayerNorm(self.cfg, self.cfg.d_mlp)
        self.dropout = nn.Dropout(self.cfg.dropout)
        self.W_in = nn.Parameter(torch.empty(self.cfg.d_model, self.cfg.d_mlp, dtype=self.cfg.dtype))
        self.b_in = nn.Parameter(torch.zeros(self.cfg.d_mlp, dtype=self.cfg.dtype))

        self.W_out = nn.Parameter(torch.empty(self.cfg.d_mlp, self.cfg.d_model, dtype=self.cfg.dtype))
        self.b_out = nn.Parameter(torch.zeros(self.cfg.d_model, dtype=self.cfg.dtype))

        self.hook_pre = HookPoint()  # [batch, pos, d_mlp]
        self.hook_mid = HookPoint()  # [batch, pos, d_mlp]
        self.hook_post = HookPoint()  # [batch, pos, d_mlp]

    def setup_hooks(self, parent_name: str):    
        self.hook_pre.name = f"{parent_name}.hook_pre"
        self.hook_mid.name = f"{parent_name}.hook_mid"
        self.hook_post.name = f"{parent_name}.hook_post"
    def forward(
        self, x: Float[torch.Tensor, "batch pos d_model"]
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        # This is equivalent to (roughly) W_in @ x + b_in. It's important to
        # use a fused addmm to ensure it matches the Huggingface implementation
        # exactly.
        pre_act = self.hook_pre(batch_addmm(self.b_in, self.W_in, x))  # [batch, pos, d_mlp]

        if (
            self.hook_mid is not None
            and self.ln is not None
        ):
            mid_act = self.hook_mid(self.act(pre_act))  # [batch, pos, d_mlp]
            post_act = self.hook_post(self.ln(mid_act))
        else:
            post_act = self.hook_post(self.act(pre_act))  # [batch, pos, d_mlp]
        return batch_addmm(self.b_out, self.W_out, post_act)

class Attention(nn.Module): 

        def __init__(self, config: GPT2MTPConfig):
            super().__init__()
            self.cfg = config

            self.W_Q = nn.Parameter(torch.empty(self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head, dtype=self.cfg.dtype))
            self.W_K = nn.Parameter(torch.empty(self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head, dtype=self.cfg.dtype))
            self.W_V = nn.Parameter(torch.empty(self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head, dtype=self.cfg.dtype))
            self.W_O = nn.Parameter(
                torch.empty(
                    self.cfg.n_heads,
                    self.cfg.d_head,
                    self.cfg.d_model,
                    dtype=self.cfg.dtype,
                ))
            self.b_Q = nn.Parameter(torch.zeros(self.cfg.n_heads, self.cfg.d_head, dtype=self.cfg.dtype))
            self.b_K = nn.Parameter(torch.zeros(self.cfg.n_heads, self.cfg.d_head, dtype=self.cfg.dtype))
            self.b_V = nn.Parameter(torch.zeros(self.cfg.n_heads, self.cfg.d_head, dtype=self.cfg.dtype))
            self.b_O = nn.Parameter(torch.zeros(self.cfg.d_model, dtype=self.cfg.dtype))

            self.attn_scale = self.cfg.attn_scale
            self.attn_dropout = nn.Dropout(self.cfg.dropout)

            self.hook_k = HookPoint()  # [batch, pos, head_index, d_head]
            self.hook_q = HookPoint()  # [batch, pos, head_index, d_head]
            self.hook_v = HookPoint()  # [batch, pos, head_index, d_head]
            self.hook_z = HookPoint()  # [batch, pos, head_index, d_head]
            self.hook_attn_scores = HookPoint()  # [batch, head_index, query_pos, key_pos]
            self.hook_pattern = HookPoint()  # [batch, head_index, query_pos, key_pos]
            self.hook_result = HookPoint()  # [batch, pos, head_index, d_model]
            self.hook_rot_k = HookPoint()
            self.hook_rot_q = HookPoint()

            sin, cos = self.calculate_sin_cos_rotary(
                self.cfg.rotary_dim,
                self.cfg.n_ctx,
                base=self.cfg.rotary_base,
                dtype=self.cfg.dtype,
            )
            # Create a max_ctx x max_ctx mask, with True iff that query position
            # can attend to that key position (query is first axis, key is second axis)
            self.causal_mask = torch.tril(torch.ones((self.cfg.n_ctx, self.cfg.n_ctx)).bool())

            self.register_buffer("rotary_sin", sin)
            self.register_buffer("rotary_cos", cos)
            self.register_buffer("mask", self.causal_mask)
            self.register_buffer("IGNORE", torch.tensor(-torch.inf))

        def setup_hooks(self, parent_name: str): 
            self.hook_q.name = f"{parent_name}.hook_q"
            self.hook_k.name = f"{parent_name}.hook_k"
            self.hook_v.name = f"{parent_name}.hook_v"
            self.hook_z.name = f"{parent_name}.hook_z"
            self.hook_attn_scores.name = f"{parent_name}.hook_attn_scores"
            self.hook_pattern.name = f"{parent_name}.hook_pattern"
            self.hook_result.name = f"{parent_name}.hook_result"
            self.hook_rot_k.name = f"{parent_name}.hook_rot_k"
            self.hook_rot_q.name = f"{parent_name}.hook_rot_q"
        
        def forward(
            self,
            query_input: Union[
                Float[torch.Tensor, "batch pos d_model"],
                Float[torch.Tensor, "batch pos head_index d_model"],
            ],
            key_input: Union[
                Float[torch.Tensor, "batch kv_pos d_model"],
                Float[torch.Tensor, "batch kv_pos head_index d_model"],
                Float[torch.Tensor, "batch kv_pos kv_head_index d_model"],
            ],
            value_input: Union[
                Float[torch.Tensor, "batch kv_pos d_model"],
                Float[torch.Tensor, "batch kv_pos head_index d_model"],
                Float[torch.Tensor, "batch kv_pos kv_head_index d_model"],
            ],
            past_kv_cache_entry: Optional[HookedTransformerKeyValueCacheEntry] = None,
            additive_attention_mask: Optional[Float[torch.Tensor, "batch 1 1 kv_pos"]] = None,
            attention_mask: Optional[Int[torch.Tensor, "batch offset_pos"]] = None,
        ) -> Float[torch.Tensor, "batch pos d_model"]:
                    
            q = self.hook_q(simple_attn_linear(query_input, self.W_Q, self.b_Q))
            k = self.hook_k(simple_attn_linear(key_input, self.W_K, self.b_K))
            v = self.hook_v(simple_attn_linear(value_input, self.W_V, self.b_V))
            if past_kv_cache_entry is not None:
            # Appends the new keys and values to the cached values, and automatically updates the cache
                kv_cache_pos_offset = past_kv_cache_entry.past_keys.size(1)
                k, v = past_kv_cache_entry.append(k, v)
            else:
                # Not using a cache
                kv_cache_pos_offset = 0

            # apply positional encoding (RoPE)
            q = self.hook_rot_q(self.apply_rotary(q, kv_cache_pos_offset, attention_mask))
            k = self.hook_rot_k(self.apply_rotary(k, 0, attention_mask)) # keys are cached so no offset
            
            if self.cfg.dtype not in [torch.float32, torch.float64]:
                # If using 16 bits, increase the precision to avoid numerical instabilities
                q = q.to(torch.float32)
                k = k.to(torch.float32)
            
            attn_scores = calculate_attn_scores(q, k, self.attn_scale) # [batch, head_index, query_pos, key_pos]
            
            # If causal attention, we mask it to only attend backwards. If bidirectional, we don't mask.
            attn_scores = apply_causal_mask(
                attn_scores, kv_cache_pos_offset, attention_mask, self.causal_mask
            )  # [batch, head_index, query_pos, key_pos]
            if additive_attention_mask is not None:
                attn_scores += additive_attention_mask

            attn_scores = self.hook_attn_scores(attn_scores)
            # Apply softmax to get attention probabilities
            pattern = F.softmax(attn_scores, dim=-1)
            pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern) # torch.Size([24, 32, 18, 18])
            pattern = self.hook_pattern(pattern)  # [batch, head_index, query_pos, key_pos]
            # Apply attention to the value matrix, i.e. individual attention head output
            v_ = einops.rearrange(
                        v, "batch key_pos head_index d_head -> batch head_index key_pos d_head"
                    )
            v_ = v_.to(self.cfg.device)
            pattern_ = einops.rearrange(
                        pattern,
                        "batch head_index query_pos key_pos -> batch head_index query_pos key_pos",
                    )
            pattern_ = pattern_.to(self.cfg.device)
            z = self.hook_z(einops.rearrange(
                            pattern_ @ v_,
                            "batch head_index query_pos d_head -> batch query_pos head_index d_head",
                    )
            ) 
            
            # Calculate the attention output as sum of the z scores
            if not self.cfg.use_attn_result:
                w = einops.rearrange(
                        self.W_O, "head_index d_head d_model -> d_model (head_index d_head)"
                    )
                out = F.linear(
                        z.reshape(z.shape[0], z.shape[1], self.cfg.d_head * self.cfg.n_heads),
                        w,
                        self.b_O,
                    )
            else: 
                # Add singleton dimensions to make shapes compatible for broadcasting:
                w = einops.rearrange(
                    self.W_O,
                    "head_index d_head d_model -> 1 1 head_index d_head d_model",
                )
                z = einops.rearrange(
                    z, "batch pos head_index d_head -> batch pos head_index d_head 1"
                )

                # Multiply the z tensor by the W_O tensor, summing over the d_head dimension
                unhooked_result = (z * w).sum(-2)  # 
                result = self.hook_result(unhooked_result)  # [batch, pos, head_index, d_model]
                # result = self.hook_result(
                #     einops.einsum(
                #         z,
                #         w,
                #         "... head_index d_head, d_model head_index d_head -> ... head_index d_model",
                #     )
                # )  # [batch, pos, head_index, d_model]
                out = (
                    einops.reduce(result, "batch position index model->batch position model", "sum")
                    + self.b_O
                )  # [batch, pos, d_model]
        
            return out
        def calculate_sin_cos_rotary(
            self,
            rotary_dim: int,
            n_ctx: int,
            base: int = 10000,
            dtype: torch.dtype = torch.float32,
        ) -> Tuple[Float[torch.Tensor, "n_ctx rotary_dim"], Float[torch.Tensor, "n_ctx rotary_dim"]]:
            """
            Calculate the sine and cosine waves to use in a rotary embedding. See https://blog.eleuther.ai/rotary-embeddings/ for details

            Note: For some inexplicable reason, in GPT-J each ADJACENT pair of elements in k and q are rotated, in GPT-NeoX the pair of elements at k and k+n//2 are rotated (ie folding the full length in half, and then looking at pairs accordingly). I have absolutely no clue why, it should be completely equivalent.
            To resolve this, I've coded it to default to the GPT-J mode, but to explicitly check whether it's GPT-NeoX and then do the GPT-NeoX thing if it is.
            """
            high_precision = torch.float32 if dtype != torch.float64 else torch.float64
            pos = torch.arange(n_ctx, dtype=high_precision)
            dim = torch.arange(rotary_dim // 2, dtype=high_precision)

            # Llama-3.1 uses NTK-by-Parts Rotary Embedding introduced in Section 3.2 in https://arxiv.org/pdf/2309.00071
            # Implementation copied from https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/modeling_rope_utils.py#L310
            if self.cfg.use_NTK_by_parts_rope:
                inv_freq = 1.0 / (
                    base ** (torch.arange(0, rotary_dim, 2, dtype=torch.int64).float() / rotary_dim)
                )
                factor = self.cfg.NTK_by_parts_factor
                low_freq_factor = self.cfg.NTK_by_parts_low_freq_factor
                high_freq_factor = self.cfg.NTK_by_parts_high_freq_factor
                old_context_len = n_ctx

                low_freq_wavelen = old_context_len / low_freq_factor
                high_freq_wavelen = old_context_len / high_freq_factor

                wavelen = 2 * math.pi / inv_freq
                inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
                smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                smoothed_inv_freq = (
                    1 - smooth_factor
                ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
                is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
                inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
                freq = 1 / inv_freq_llama
            else:
                freq = base ** (dim / (rotary_dim / 2))
            if self.cfg.rotary_adjacent_pairs:
                freq = einops.repeat(freq, "d -> (d 2)")
            else:
                freq = einops.repeat(freq, "d -> (2 d)")
            # Create a n_ctx x rotary_dim tensor, where each column is an arithmetic sequence of angles in that frequency
            angles = pos[:, None] / freq[None, :]
            return torch.sin(angles).to(dtype), torch.cos(angles).to(dtype)

        def rotate_every_two(
            self, x: Float[torch.Tensor, "... rotary_dim"]
        ) -> Float[torch.Tensor, "... rotary_dim"]:
            """
            Rotary helper function, splits x into blocks of size 2 along the final axis and maps [x0, x1] to [-x1, x0]

            The final axis of x must have even length.

            GPT-NeoX and GPT-J do rotary subtly differently, see calculate_sin_cos_rotary for details.
            """
            rot_x = x.clone()
            if self.cfg.rotary_adjacent_pairs:
                rot_x[..., ::2] = -x[..., 1::2]
                rot_x[..., 1::2] = x[..., ::2]
            else:
                n = x.size(-1) // 2
                rot_x[..., :n] = -x[..., n:]
                rot_x[..., n:] = x[..., :n]

            return rot_x

        def apply_rotary(
            self,
            x: Float[torch.Tensor, "batch pos head_index d_head"],
            past_kv_pos_offset=0,
            attention_mask: Optional[Int[torch.Tensor, "batch offset_pos"]] = None,
        ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
            # Only apply rotary to first rotary_dim dimensions (eg, if rotary_dim=64 and d_head=256, only apply to first 1/4 of dimensions)

            if x.device != self.rotary_sin.device:
                x = x.to(self.rotary_sin.device)

            x_pos = x.size(1)
            x_rot = x[..., : self.cfg.rotary_dim]
            x_pass = x[..., self.cfg.rotary_dim :]
            x_flip = self.rotate_every_two(x_rot)

            if attention_mask is None:
                rotary_cos = self.rotary_cos[
                    None, past_kv_pos_offset : past_kv_pos_offset + x_pos, None, :
                ]
                rotary_sin = self.rotary_sin[
                    None, past_kv_pos_offset : past_kv_pos_offset + x_pos, None, :
                ]
                x_rotated = x_rot * rotary_cos + x_flip * rotary_sin
            else:
                offset_position_ids = get_offset_position_ids(past_kv_pos_offset, attention_mask)
                offset_position_ids = offset_position_ids.to(self.rotary_cos.device)
                mask_rotary_cos = self.rotary_cos[offset_position_ids, None, :]
                mask_rotary_sin = self.rotary_sin[offset_position_ids, None, :]
                x_rotated = x_rot * mask_rotary_cos + x_flip * mask_rotary_sin

            return torch.cat([x_rotated, x_pass], dim=-1)            
