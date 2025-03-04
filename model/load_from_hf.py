import os 
import torch
from typing import Dict
from transformers import AutoModelForCausalLM, AutoConfig
from transformer_lens.pretrained.weight_conversions import convert_gpt2_weights
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

def get_pretrained_state_dict(
    official_model_name: str,
    cfg: HookedTransformerConfig,
    hf_model=None,
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    huggingface_token = os.environ.get("HF_TOKEN", None)
    hf_model = AutoModelForCausalLM.from_pretrained(
                        official_model_name,
                        torch_dtype=dtype,
                        token=huggingface_token,
                        **kwargs,
                    )
    state_dict = convert_gpt2_weights(hf_model, cfg)
    return state_dict

def convert_hf_model_config(model_name: str, **kwargs):
    huggingface_token = os.environ.get("HF_TOKEN", None)
    hf_config = AutoConfig.from_pretrained(
                official_model_name,
                token=huggingface_token,
                **kwargs,
            )
    architecture = model_name.split("-")[0]
    official_model_name = model_name
    cfg_dict = {
                "d_model": hf_config.n_embd,
                "d_head": hf_config.n_embd // hf_config.n_head,
                "n_heads": hf_config.n_head,
                "d_mlp": hf_config.n_embd * 4,
                "n_layers": hf_config.n_layer,
                "n_ctx": hf_config.n_ctx,
                "eps": hf_config.layer_norm_epsilon,
                "d_vocab": hf_config.vocab_size,
                "act_fn": hf_config.activation_function,
                "use_attn_scale": True,
                "use_local_attn": False,
                "scale_attn_by_inverse_layer_idx": hf_config.scale_attn_by_inverse_layer_idx,
                "normalization_type": "LN",
            }
    
    # All of these models use LayerNorm
    cfg_dict["original_architecture"] = architecture
    # The name such that AutoTokenizer.from_pretrained works
    cfg_dict["tokenizer_name"] = official_model_name
    if kwargs.get("trust_remote_code", False):
        cfg_dict["trust_remote_code"] = True
    return cfg_dict

