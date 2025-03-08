"""
Configuration for GPT2MTP model, some are taken from HookedTransformerConfig class in transformer_lens.
"""
import math
from dataclasses import dataclass
from typing import Optional
import torch
import numpy as np
import random

@dataclass
class GPT2MTPConfig():
    """Configuration for GPT2MTP model."""
    d_vocab: int = 50257
    n_ctx: int = 256 # block_size + n_future ≤ max_context_length (1024 for all GPT-2's // 8)
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_mlp: int = 3072
    d_head: int = 64
    d_vocab_out: int = 50257
    attn_scale: float = math.sqrt(d_head)
    use_attn_result: bool = False
    use_split_qkv_input: bool = False
    n_key_value_heads: Optional[int] = None
    dropout: float = 0.0
    bias: bool = True
    dtype: str = torch.float32
    post_embedding_ln: bool = False
    act_function: str = "gelu"
    rotary_adjacent_pairs: bool = False
    rotary_dim: Optional[int] = 64
    rotary_base: int = 10000
    use_NTK_by_parts_rope: bool = False
    NTK_by_parts_low_freq_factor: float = 1.0
    NTK_by_parts_high_freq_factor: float = 4.0
    NTK_by_parts_factor: float = 8.0
    mtp_heads: int = 4
    loss_type: str = "multi_token"
    tokenizer_name: str = "gpt2"
    default_prepend_bos: bool = True
    device: Optional[str] = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_devices: int = 1
    iniitalizer_range: float = -1.0

    def __post_init__(self):
        if self.iniitalizer_range < 0:
            # Roughly copy the GPT-2 value, but proportional to sqrt(1/d_model)
            self.initializer_range = 0.8 / np.sqrt(self.d_model)
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    def set_seed_everywhere(self, seed: int):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        