"""GPT2MTP base classes.

 contains the TransformerBlock and the GPT2MTP Class. Some are taken from transformer_lens library
"""

import logging
from jaxtyping import Float, Int 
import os
import numpy as np
import torch
from typing import Dict, List, NamedTuple, Optional, Tuple, Union, overload
from typing_extensions import Literal

from torch import nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from model.config import GPT2MTPConfig
from model.utils_mtp import (
    lm_cross_entropy_loss,
    mtp_cross_entropy_loss,
    sample_top_p
    )

from model.model_components import (
    Attention,
    MLP,
    LayerNorm, 
)

from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.utils import repeat_along_head_dimension
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache, HookedTransformerKeyValueCacheEntry
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.utilities import devices
from transformer_lens.components import (
    Embed,
    Unembed
)
import transformer_lens.utils as utils

USE_DEFAULT_VALUE = None
SingleLoss = Float[torch.Tensor, ""]  # Type alias for a single element tensor
LossPerToken = Float[torch.Tensor, "batch pos-1"]
Loss = Union[SingleLoss, LossPerToken]

class Output(NamedTuple):
    """Output Named Tuple.

    Named tuple object for if we want to output both logits and loss.
    """

    logits: Float[torch.Tensor, "batch pos d_vocab"]
    loss: Loss
    attention_mask: Optional[Int[torch.Tensor, "batch pos"]]


class TransformerBlock(nn.Module):
    def __init__(self, config: GPT2MTPConfig):
        super().__init__()
        self.cfg = config
        self.ln1 = LayerNorm(self.cfg)
        self.ln2 = LayerNorm(self.cfg)
        self.attn = Attention(self.cfg)
        self.mlp = MLP(self.cfg)

        # Define hook points for intermediate activations:
        self.hook_attn_in = HookPoint() # [batch, pos, n_heads, d_model]
        self.hook_q_input = HookPoint() # [batch, pos, n_heads, d_model]
        self.hook_k_input = HookPoint() # [batch, pos, n_heads, d_model]
        self.hook_v_input = HookPoint() # [batch, pos, n_heads, d_model]
        self.hook_mlp_in = HookPoint()  # [batch, pos, n_heads, d_model]
        
        self.hook_attn_out = HookPoint() # [batch, pos, d_model]
        self.hook_mlp_out = HookPoint() # [batch, pos, d_model]
        self.hook_resid_pre = HookPoint() # [batch, pos, d_model]
        self.hook_resid_mid = HookPoint() # [batch, pos, d_model]
        self.hook_resid_post = HookPoint() # [batch, pos, d_model]

        # # Optionally, add the MTP heads in this block.
        # self.add_mtp_heads = add_mtp_heads
        # if self.add_mtp_heads:
        #     # Create n_future linear projections (one per future token)
        #     self.mtp_heads = nn.ModuleList([
        #         nn.Linear(self.cfg.d_model, self.cfg.d_model, bias=self.cfg.bias)
        #         for _ in range(n_future)
        #     ])
        #     # HookPoint for the stacked outputs from all MTP heads pre/post LayerNorm
        #     self.hook_mtp_heads_pre = HookPoint()
        #     self.hook_mtp_heads_post = HookPoint()
    
    def setup_hooks(self, parent_name: str):
        self.attn.setup_hooks(parent_name + ".attn")
        self.mlp.setup_hooks(parent_name + ".mlp")
        self.hook_attn_in.name = f"{parent_name}.hook_attn_in"
        self.hook_q_input.name = f"{parent_name}.hook_q_input"
        self.hook_k_input.name = f"{parent_name}.hook_k_input"
        self.hook_v_input.name = f"{parent_name}.hook_v_input"
        self.hook_mlp_in.name = f"{parent_name}.hook_mlp_in"
        self.hook_attn_out.name = f"{parent_name}.hook_attn_out"
        self.hook_mlp_out.name = f"{parent_name}.hook_mlp_out"
        self.hook_resid_pre.name = f"{parent_name}.hook_resid_pre"
        self.hook_resid_mid.name = f"{parent_name}.hook_resid_mid"
        self.hook_resid_post.name = f"{parent_name}.hook_resid_post"

    def forward(
        self,
        resid_pre: Float[torch.Tensor, "batch pos d_model"],
        past_kv_cache_entry: Optional[HookedTransformerKeyValueCacheEntry] = None,
        attention_mask: Optional[Int[torch.Tensor, "batch offset_pos"]] = None,
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        resid_pre = self.hook_resid_pre(resid_pre)  # [batch, pos, d_model]
        
        if self.cfg.use_split_qkv_input: 
            # Split the input into separate query, key, and value inputs.
            query_input = self.hook_q_input(
                repeat_along_head_dimension(resid_pre, n_heads=self.cfg.n_heads)
            )
            key_input = self.hook_k_input(
                repeat_along_head_dimension(resid_pre, n_heads=self.cfg.n_heads)
            )
            value_input = self.hook_v_input(
                repeat_along_head_dimension(resid_pre, n_heads=self.cfg.n_heads)
            )
        else: 
            attn_in = resid_pre
            query_input = attn_in
            key_input = attn_in
            value_input = attn_in
        # hook the residual stream states that are used to calculate the
        # queries, keys and values, independently.
        # Then take the layer norm of these inputs, and pass these to the attention module.
        attn_out = self.attn(
            query_input=self.ln1(query_input), 
            key_input=self.ln1(key_input), 
            value_input=self.ln1(value_input),
            past_kv_cache_entry=past_kv_cache_entry, 
            attention_mask=attention_mask
        )
        attn_out = self.hook_attn_out(attn_out)
        resid_mid = self.hook_resid_mid(resid_pre + attn_out)
        mlp_in = self.hook_mlp_in(resid_mid)
        normalized_resid_mid = self.ln2(mlp_in)
        mlp_out = self.mlp(normalized_resid_mid)
        mlp_out = self.hook_mlp_out(mlp_out)
        resid_post = self.hook_resid_post(resid_mid + mlp_out)
        return resid_post
    
class GPT2MTP(HookedRootModule):
    def __init__(self, 
                 config: GPT2MTPConfig,
                 tokenizer: Optional[object] = None, 
                 default_padding_side: Literal["left", "right"] = "right",
                 n_future: int = 4,
                ):
        super().__init__()
        self.cfg = config
        self.n_future = n_future
        
        if tokenizer is not None:
            self.tokenizer = self.set_tokenizer(tokenizer, default_padding_side=default_padding_side)
        else: 
            huggingface_token = os.environ.get("HF_TOKEN", None)
            use_fast = True 
            self.set_tokenizer(
                AutoTokenizer.from_pretrained(
                            self.cfg.tokenizer_name,
                            add_bos_token=True,
                            trust_remote_code=False,
                            use_fast=use_fast,
                            token=huggingface_token,
                        ),
                        default_padding_side=default_padding_side,
                    )
        self.embed = Embed(self.cfg)      # Embedding + positional embedding if included inside
        self.hook_embed = HookPoint()
        self.ln_f = LayerNorm(self.cfg)
        self.unembed = Unembed(self.cfg)  # Shared linear layer mapping d_model -> vocab_size
        self.hook_unembed = HookPoint()
        
        # MTP paper implementation uses a single linear layer for the MTP heads with a shared trunk from the last layer residual.
        # Create n_future independent prediction heads.  
        # Each head projects the final hidden state into the intermediate representation of the residual stream of the next token <= n_future.
        self.blocks = nn.ModuleList(
            [TransformerBlock(self.cfg) for _ in range(self.cfg.n_layers)]
        )
        self.hook_mtp_heads = nn.ModuleList([
            HookPoint() for _ in range(self.n_future)
        ])
        # Create MTP heads with individual hooks
        self.mtp_heads = nn.ModuleList([
            self.hook_mtp_heads[i](nn.Linear(self.cfg.d_model, self.cfg.d_model, bias=self.cfg.bias))
            for i in range(self.n_future)
        ])
        self.hook_mtp_heads_out_pre = HookPoint()
        self.hook_mtp_heads_out_post = HookPoint()

        # DeepseekV3 uses transformer blocks for the MTP modules
        # Create transformer blocks, and only add MTP heads on the final block.
        # self.blocks = torch.nn.ModuleList()
        # for layer_id in range(self.cfg.n_layers - n_future + 1):
        #         block = TransformerBlock(self.cfg)
        #         block.setup_hooks(f"blocks.{layer_id}")  # Correct naming
        #         # block.name = f"blocks.{layer_id}"
        #         self.blocks.append(block)
        # # NEW: works but needs to be adapted to DeepseekV3 
        # self.add_mtp_heads = nn.ModuleList()
        # for l in range(len(self.blocks) - self.n_future + 1, len(self.blocks)):
        #     # self.add_mtp_heads.append(TransformerBlock(self.cfg))
        #     block = TransformerBlock(self.cfg)
        #     block.setup_hooks(f"mtp_heads.{l}")  # Correct naming
        #     self.add_mtp_heads.append(block)
        #     self.hook_mtp_heads_out_pre = HookPoint()
        #     self.hook_mtp_heads_out_post = HookPoint()
        # # OLD
        # # Optionally, add the MTP heads in this block.
        # if self.add_mtp_heads:
        #     # Create n_future linear projections (one per future token)
        #     self.mtp_heads = nn.ModuleList([
        #         nn.Linear(self.cfg.d_model, self.cfg.d_model, bias=self.cfg.bias)
        #         for _ in range(n_future)
        #     ])
        #     # HookPoint for the stacked outputs from all MTP heads pre/post LayerNorm
        #     self.hook_mtp_heads_pre = HookPoint()
        #     self.hook_mtp_heads_post = HookPoint()


        # Gives each module a parameter with its name (relative to this root module)
        # Needed for HookPoints to work
        self.setup()

    def forward(
        self,
        input: Union[
            str,
            List[str],
            Int[torch.Tensor, "batch pos"],
            Float[torch.Tensor, "batch pos d_model"],
        ],
        targets: Optional[Int[torch.Tensor, "batch pos + n_future"]] = None,
        return_type: Optional[str] = "logits",
        # return_all_heads: bool = False,
        loss_type: str = "multi_token",
        loss_per_token: bool = False,
        prepend_bos: Optional[Union[bool, None]] = None,
        padding_side: Optional[Literal["left", "right"]] = None,
        start_at_layer: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,  # [batch pos]
        stop_at_layer: Optional[int] = None,
        past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
    ):
        """
        Forward pass for multi-token prediction.
        x: (batch, seq) input token IDs.
        targets: (batch, seq, n_future) target token IDs for each future token.
                 If provided, the loss is computed.
        Returns:
          logits: (batch, seq, n_future, vocab_size)
          loss (if targets is provided), averaged over future steps.
        """
        # Embed tokens
        if start_at_layer is None:
            (
                residual,
                tokens,
                attention_mask,
            ) = self.input_to_embed(
                input,
                prepend_bos=prepend_bos,
                padding_side=padding_side,
                attention_mask=attention_mask,
                past_kv_cache=past_kv_cache,
            )
        else:
            assert type(input) == torch.Tensor
            residual = input

        if start_at_layer is None:
            start_at_layer = 0
        # If we explicitly want to start or stop at a layer, we only iterate through the blocks
        # between those indices. Note that start_at_layer is inclusive and stop_at_layer is
        # exclusive.
        # Eg: start_at_layer==None + stop_at_layer==0 means to only run the embed.
        # Eg: start_at_layer==3 + stop_at_layer==-1 means to run from layer 3 until the end of the PENULTIMATE layer
        # We stop at the final block (7) up until the MTP heads start in order to retrieve the shared trunk output.
        blocks_and_idxs = list(zip(range(self.cfg.n_layers), self.blocks))
        for i, block in blocks_and_idxs[start_at_layer:stop_at_layer]:  # type: ignore
            # Note that each block includes skip connections, so we don't need
            # residual + block(residual)
            # If we're using multiple GPUs, we need to send the residualto the correct GPU
            residual = residual.to(devices.get_device_for_block_index(i, self.cfg))
            
            residual = block(
                residual,
                # Cache contains a list of HookedTransformerKeyValueCache objects, one for each
                # block
                past_kv_cache_entry=past_kv_cache[i] if past_kv_cache is not None else None,
                attention_mask=attention_mask,
            )  # [batch, pos, d_model]

        if stop_at_layer is not None:
            # We retrieve the shared trunk output from the final block before the MTP heads start.
            return residual
        
        if self.mtp_heads is not None:
            mtp_outputs = []
            # Compute outputs for each of the n_future MTP heads.
            for i, head in enumerate(self.mtp_heads):
                mtp_head = self.hook_mtp_heads[i](head(residual))
                mtp_outputs.append(mtp_head)    
            # Stack the outputs along a new dimension (n_future)
            stacked_pre = self.hook_mtp_heads_out_pre(torch.stack(mtp_outputs, dim=-2))  # (batch, pos, n_future, d_model)
            # Pass the stacked tensor through a LayerNorm.
            stacked_norm = self.hook_mtp_heads_out_post(self.ln_f(stacked_pre))  # (batch, pos, n_future, d_model)
        # Map the hooked output to vocabulary logits using the shared unembedding layer
        logits = self.hook_unembed(self.unembed(stacked_norm))  # Shape: (batch, seq, n_future, vocab_size)
        # Compute cross-entropy loss for each future token prediction.
        if return_type == "logits":
                return logits
        else:
            assert (
                tokens is not None
            ), "tokens must be passed in if return_type is 'loss' or 'both'"
            if loss_type == "multi_token":
                loss = self.loss_fn(logits, targets, attention_mask, per_token=loss_per_token, loss_type=loss_type)
            else: 
                loss = self.loss_fn(logits, targets, attention_mask, per_token=loss_per_token, loss_type='single_token')
            if return_type == "loss":
                return loss
            elif return_type == "both":
                return Output(logits, loss, attention_mask)
            else:
                logging.warning(f"Invalid return_type passed in: {return_type}")
                return None
            
    def loss_fn(
        self,
        logits: Float[torch.Tensor, "batch pos d_vocab"],
        targets: Int[torch.Tensor, "batch pos"],
        attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
        per_token: bool = False,
        loss_type: Literal["single_token", "multi_token"] = "single_token",
    ):
        """Wrapper around `utils.lm_cross_entropy_loss`.

        Used in forward() with return_type=="loss" or "both".
        """    
        targets = targets.to(logits.device)
        if loss_type == "single_token":
            return lm_cross_entropy_loss(logits, targets, attention_mask, per_token)
        elif loss_type == "multi_token":
            return mtp_cross_entropy_loss(logits, targets, attention_mask, per_token)
        
    
    def input_to_embed(
        self,
        input: Union[str, List[str], Int[torch.Tensor, "batch pos"]],
        prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
        padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
    ) -> Tuple[
        Float[torch.Tensor, "batch pos d_model"],  # residual
        Optional[Int[torch.Tensor, "batch pos"]],  # tokens
        Optional[Float[torch.Tensor, "batch pos d_model"]],  # shortformer_pos_embed
        Optional[torch.Tensor],  # attention_mask [batch pos]
    ]:
        """Convert input to first residual stream.

        Args:
            input (Union[str, List[str], Int[torch.Tensor, "batch pos"]]): The input to the model.
            prepend_bos (bool, optional): Overrides self.cfg.default_prepend_bos. Whether to prepend
                the BOS token to the input (only applies when input is a string). Defaults to None,
                implying usage of self.cfg.default_prepend_bos which is set to True unless specified
                otherwise. Pass True or False to locally override the default.
            padding_side ([Literal["left", "right"], optional): Overrides
                self.tokenizer.padding_side. Specifies which side to pad when tokenizing
                multiple strings of different lengths.
            past_kv_cache (HookedTransformerKeyValueCache, optional): If passed, we're doing caching
                and attention_mask will be stored in the cache.
        """
        if isinstance(input, str) or isinstance(input, list):
            # If text, convert to tokens (batch_size=1)
            assert (
                self.tokenizer is not None
            ), "Must provide a tokenizer if passing a string to the model"
            # This is only intended to support passing in a single string
            tokens = self.to_tokens(input, prepend_bos=prepend_bos, padding_side=padding_side)
        else:
            tokens = input
        if len(tokens.shape) == 1:
            # If tokens are a rank 1 tensor, add a dummy batch dimension to avoid things breaking.
            tokens = tokens[None]
        if tokens.device.type != self.cfg.device:
            tokens = tokens.to(devices.get_device_for_block_index(0, self.cfg))

        if attention_mask is not None:
            assert attention_mask.shape == tokens.shape, (
                f"Attention mask shape {attention_mask.shape} does not match tokens shape "
                f"{tokens.shape}"
            )
            attention_mask = attention_mask.to(devices.get_device_for_block_index(0, self.cfg))
        elif (
            self.tokenizer and self.tokenizer.padding_side == "left"
        ) or past_kv_cache is not None:
            # If the padding side is left or we are using caching, we need to compute the attention
            # mask for the adjustment of absolute positional embeddings and attention masking so
            # that pad tokens are not attended.

            if prepend_bos is USE_DEFAULT_VALUE:
                prepend_bos = self.cfg.default_prepend_bos
            attention_mask = utils.get_attention_mask(self.tokenizer, tokens, prepend_bos)

            if past_kv_cache is not None:
                # past_kv_cache is not None, so we're doing caching.
                # We need to extend the previous attention_mask.
                # Update the past_kv_cache with the new attention_mask (unless it's frozen)
                attention_mask = past_kv_cache.append_attention_mask(attention_mask)
        else:
            # We separate this case from for computational efficiency.
            attention_mask = None

        # If we're doing caching, then we reuse keys and values from previous runs, as that's the
        # only way that past activations will affect the final logits. The cache contains those so
        # we don't need to recompute them. This is useful for generating text. As we have absolute
        # positional encodings, to implement this we have a `pos_offset` variable, defaulting to
        # zero, which says to offset which positional encodings are used (cached keys and values
        # were calculated with their own positional encodings).
        if past_kv_cache is None:
            pos_offset = 0
        else:
            batch_size, ctx_length = tokens.shape
            (
                cached_batch_size,
                cache_ctx_length,
                num_heads_in_cache,
                d_head_in_cache,
            ) = past_kv_cache[0].past_keys.shape
            assert cached_batch_size == batch_size
            if self.cfg.n_key_value_heads is None:
                assert num_heads_in_cache == self.cfg.n_heads
            else:
                assert num_heads_in_cache == self.cfg.n_key_value_heads
            assert d_head_in_cache == self.cfg.d_head
            pos_offset = cache_ctx_length
        # if self.cfg.use_hook_tokens:
        #     tokens = self.hook_tokens(tokens)
        embed = self.hook_embed(self.embed(tokens))  # [batch, pos, d_model]
        # Rotary doesn't use positional embeddings, instead they're applied when dot producting
        # keys and queries. 
        residual = embed
        return residual, tokens,  attention_mask
    
    def shared_trunk(
        self,
        x: torch.Tensor,
        prepend_bos: Optional[bool] = None,
        padding_side: Optional[Literal["left", "right"]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv_cache: Optional[HookedTransformerKeyValueCache] = None,
        start_at_layer: Optional[int] = None,
        stop_at_layer: Optional[int] = None,
    ) -> torch.Tensor:
            """
            Processes input x through the shared trunk of transformer blocks.
            
            Args:
                x (torch.Tensor): Input token IDs with shape (batch, seq).
                prepend_bos (bool, optional): Whether to prepend a BOS token.
                padding_side (str, optional): Padding side ("left" or "right").
                attention_mask (torch.Tensor, optional): Attention mask of shape (batch, seq).
                past_kv_cache (optional): Past key-value cache (if using caching).
                start_at_layer (int, optional): Optionally start at a specific block index.
                stop_at_layer (int, optional): Optionally stop at a specific block index.
            
            Returns:
                torch.Tensor: The shared trunk output of shape (batch, seq, d_model).
            """
            # Convert the input tokens into embeddings, return the first residual and attention mask.
            if start_at_layer is None:
                residual, _, attention_mask = self.input_to_embed(
                    x,
                    prepend_bos=prepend_bos,
                    padding_side=padding_side,
                    attention_mask=attention_mask,
                    past_kv_cache=past_kv_cache,
                )
            else:
                # If x is already a tensor (e.g. from a previous layer), we use it directly.
                residual = x

            if start_at_layer is None:
                start_at_layer = 0

            # Process through the transformer trunk blocks.
            for i, block in enumerate(self.blocks[start_at_layer:stop_at_layer], start=start_at_layer):
                # If using caching, you can pass the corresponding past_kv_cache entry.
                kv_entry = past_kv_cache[i] if past_kv_cache is not None else None
                residual = block(residual, past_kv_cache_entry=kv_entry, attention_mask=attention_mask)

            # Apply the final layer normalization.
            trunk_output = self.ln_f(residual)
            return trunk_output

    def set_tokenizer(
        self,
        tokenizer,
        default_padding_side="right",
    ):
        """Set the tokenizer to use for this model.

        Args:
            tokenizer (PreTrainedTokenizer): a pretrained HuggingFace tokenizer.
            default_padding_side (str): "right" or "left", which side to pad on.

        """
        assert isinstance(
            tokenizer, PreTrainedTokenizerBase
        ), f"{type(tokenizer)} is not a supported tokenizer, please use PreTrainedTokenizer or PreTrainedTokenizerFast"

        assert default_padding_side in [
            "right",
            "left",
        ], f"padding_side must be 'right' or 'left', got {default_padding_side}"

        # Use a tokenizer that is initialized with add_bos_token=True as the default tokenizer.
        # Such a tokenizer should be set as the default tokenizer because the tokenization of some
        # tokenizers like LlamaTokenizer are different when bos token is automatically/manually
        # prepended, and add_bos_token cannot be dynamically controlled after initialization
        # (https://github.com/huggingface/transformers/issues/25886).
        tokenizer_with_bos = utils.get_tokenizer_with_bos(tokenizer)
        self.tokenizer = tokenizer_with_bos
        assert self.tokenizer is not None  # keep mypy happy
        self.tokenizer.padding_side = default_padding_side

        # Some tokenizers doesn't automatically prepend the BOS token even when they are initialized
        # with add_bos_token=True. Therefore, we need this information to dynamically control prepend_bos.
        self.cfg.tokenizer_prepends_bos = len(self.tokenizer.encode("")) > 0

        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = "<|endoftext|>"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token

        # Infer vocab size from tokenizer
        if self.cfg.d_vocab == -1:
            self.cfg.d_vocab = max(self.tokenizer.vocab.values()) + 1
        if self.cfg.d_vocab_out == -1:
            self.cfg.d_vocab_out = self.cfg.d_vocab

    def to_tokens(
            self,
            input: Union[str, List[str]],
            prepend_bos: Optional[Union[bool, None]] = USE_DEFAULT_VALUE,
            padding_side: Optional[Union[Literal["left", "right"], None]] = USE_DEFAULT_VALUE,
            move_to_device: bool = True,
            truncate: bool = True,
        ) -> Int[torch.Tensor, "batch pos"]:
            """Converts a string to a tensor of tokens.

            If prepend_bos is True, prepends the BOS token to the input - this is recommended when
            creating a sequence of tokens to be input to a model.

            Gotcha: prepend_bos prepends a beginning of string token. This is a recommended default when
            inputting a prompt to the model as the first token is often treated weirdly, but should only
            be done at the START of the prompt. Make sure to turn it off if you're looking at the
            tokenization of part of the prompt! (Note: some models eg GPT-2 were not trained with a BOS
            token, others (OPT and my models) were)

            Gotcha2: Tokenization of a string depends on whether there is a preceding space and whether
            the first letter is capitalized. It's easy to shoot yourself in the foot here if you're not
            careful!

            Args:
                input (Union[str, List[str]]): The input to tokenize.
                prepend_bos (bool, optional): Overrides self.cfg.default_prepend_bos. Whether to prepend
                    the BOS token to the input (only applies when input is a string). Defaults to None,
                    implying usage of self.cfg.default_prepend_bos which is set to True unless specified
                    otherwise. Pass True or False to locally override the default.
                padding_side (Union[Literal["left", "right"], None], optional): Overrides
                    self.tokenizer.padding_side. Specifies which side to pad when tokenizing
                    multiple strings of different lengths.
                move_to_device (bool): Whether to move the output tensor of tokens to the device the
                    model lives on. Defaults to True truncate (bool): If the output tokens are too long,
                    whether to truncate the output tokens to the model's max context window. Does nothing
                    for shorter inputs. Defaults to True.
            """
            with utils.LocallyOverridenDefaults(
                self, prepend_bos=prepend_bos, padding_side=padding_side
            ):
                assert self.tokenizer is not None, "Cannot use to_tokens without a tokenizer"
                assert (
                    self.cfg.tokenizer_prepends_bos is not None
                ), "Set the tokenizer for the model by calling set_tokenizer"

                if self.cfg.default_prepend_bos and not self.cfg.tokenizer_prepends_bos:
                    # We want to prepend bos but the tokenizer doesn't automatically do it, so we add it manually
                    input = utils.get_input_with_manually_prepended_bos(self.tokenizer, input)

                tokens = self.tokenizer(
                    input,
                    return_tensors="pt",
                    padding=True,
                    truncation=truncate,
                    max_length=self.cfg.n_ctx if truncate else None,
                )["input_ids"]

                if not self.cfg.default_prepend_bos and self.cfg.tokenizer_prepends_bos:
                    # We don't want to prepend bos but the tokenizer does it automatically, so we remove it manually
                    tokens = utils.get_tokens_with_bos_removed(self.tokenizer, tokens)

                if move_to_device:
                    tokens = tokens.to(self.cfg.device)
                return tokens

    def to_string(
        self,
        tokens: Union[
            List[int],
            Int[torch.Tensor, ""],
            Int[torch.Tensor, "batch pos"],
            Int[torch.Tensor, "pos"],
            np.ndarray,
            List[Int[torch.Tensor, "pos"]],
        ],
    ) -> Union[str, List[str]]:
        """Tokens to String(s).

        Converts a tensor of tokens to a string (if rank 1) or a list of strings (if rank 2).

        Accepts lists of tokens and numpy arrays as inputs too (and converts to tensors internally)
        """
        assert self.tokenizer is not None, "Cannot use to_string without a tokenizer"

        if not isinstance(tokens, torch.Tensor):
            # We allow lists to be input
            tokens = torch.tensor(tokens)

        # I'm not sure what exactly clean_up_tokenization_spaces does, but if
        # it's set, then tokenization is no longer invertible, and some tokens
        # with a bunch of whitespace get collapsed together
        if len(tokens.shape) == 2:
            return self.tokenizer.batch_decode(tokens, clean_up_tokenization_spaces=False)
        elif len(tokens.shape) <= 1:
            return self.tokenizer.decode(tokens, clean_up_tokenization_spaces=False)
        else:
            raise ValueError(f"Invalid shape passed in: {tokens.shape}")
        
    def init_weights(self):
        """
        Initialize weights with GPT-2 initialization. Biases are initialized to 0.0 and weights
        are initialized to N(0, 0.64/d_model) if initializer_range is not set, otherwise std is initializer_range.
        """
        for name, param in self.named_parameters():
            if "W_" in name:
                nn.init.normal_(param, std=self.cfg.initializer_range)
            elif "b_" in name:
                nn.init.constant_(param, 0.0)
    
    @overload
    def run_with_cache(
        self, *model_args, return_cache_object: Literal[True] = True, **kwargs
    ) -> Tuple[Output, ActivationCache]:
        ...

    @overload
    def run_with_cache(
        self, *model_args, return_cache_object: Literal[False], **kwargs
    ) -> Tuple[Output, Dict[str, torch.Tensor]]:
        ...

    def run_with_cache(
        self, *model_args, return_cache_object=True, remove_batch_dim=False, **kwargs
    ) -> Tuple[
        Union[
            None,
            Float[torch.Tensor, "batch pos d_vocab"],
            Loss,
            Tuple[Float[torch.Tensor, "batch pos d_vocab"], Loss],
        ],
        Union[ActivationCache, Dict[str, torch.Tensor]],
    ]:
        """Wrapper around `run_with_cache` in HookedRootModule.

        If return_cache_object is True, this will return an ActivationCache object, with a bunch of
        useful HookedTransformer specific methods, otherwise it will return a dictionary of
        activations as in HookedRootModule.
        """
        out, cache_dict = super().run_with_cache(
            *model_args, remove_batch_dim=remove_batch_dim, **kwargs
        )
        if return_cache_object:
            cache = ActivationCache(cache_dict, self, has_batch_dim=not remove_batch_dim)
            return out, cache
        else:
            return out, cache_dict
        
    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        do_sample: bool = True,
        stop_at_eos: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> List[str]:
        """
        Generate text sequences using multi-token prediction.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts.
            max_gen_len (int): Maximum new tokens to generate.
            temperature (float, optional): Sampling temperature.
            top_p (float, optional): Nucleus sampling threshold.
            do_sample (bool, optional): Whether to sample (if False, use argmax).
            verbose (bool, optional): If True, show progress.
            stop_at_eos (bool, optional): Stop generation when EOS is produced.
            eos_token_id (Optional[int], optional): EOS token ID; if None, use self.tokenizer.eos_token_id.

        Returns:
            List[str]: Generated texts for each prompt.
        """
        # Prepare prompt tokens: pad to same length
        prompt_tokens = self.to_tokens(prompt_tokens, prepend_bos=False, padding_side="right")
        bsz = len(prompt_tokens)
        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        total_len = max_prompt_len  # initial prompt length; generated tokens will be appended

        pad_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, "pad_token_id") else 0
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=self.cfg.device)
        for i, t in enumerate(prompt_tokens):
            tokens[i, : len(t)] = torch.as_tensor(t, dtype=torch.long, device=self.cfg.device)

        # Optionally initialize past kv cache.
        use_past_kv_cache = True  # set based on your design
        if use_past_kv_cache:
            past_kv_cache = HookedTransformerKeyValueCache.init_cache(self.cfg, self.cfg.device, bsz)
        else:
            past_kv_cache = None

        finished = torch.zeros(bsz, dtype=torch.bool, device=self.cfg.device)
        self.eval()
        # Generation loop: we'll generate up to max_gen_len new tokens.
        for cur_gen in range(max_gen_len):
            # Use caching if available: for subsequent steps, only feed in the last token.
            if use_past_kv_cache:
                if cur_gen == 0:
                    out = self.forward(tokens, return_type="logits", loss_type="single_token", past_kv_cache=past_kv_cache)
                else:
                    out = self.forward(tokens[:, -1:], return_type="logits", loss_type="single_token", past_kv_cache=past_kv_cache)
            else:
                out = self.forward(tokens, return_type="logits", loss_type="single_token")
            # out shape: (batch, seq, n_future, vocab_size)
            # For standard generation, select the prediction from the first head (i.e. head index 0).
            logits = out.squeeze(2) if out.shape[2] == 1 else out[:, -1, 0, :]  # shape: (batch, vocab_size)
            # Alternatively, if out.shape[2]>1, you can choose to average or select head 0.
            # Sample next token:
            if do_sample and temperature > 0:
                # Compute probabilities with temperature scaling.
                probs = torch.softmax(logits / temperature, dim=-1)
                # Use nucleus (top-p) sampling (assume sample_top_p is implemented)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            # Append sampled token to tokens.
            next_token = next_token.view(tokens.size(0), 1)
            tokens = torch.cat([tokens, next_token], dim=-1)
            # Optionally mark sequences as finished if they generate EOS.
            if stop_at_eos:
                eos = eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id
                finished |= (next_token.squeeze(-1) == eos)
                if finished.all():
                    break

        # Decode the generated tokens
        outputs = []
        for i in range(bsz):
            # Remove padding
            gen_tokens = tokens[i].tolist()
            # Optionally, remove prompt tokens.
            prompt_len = len(prompt_tokens[i])
            gen_tokens = gen_tokens[prompt_len:]
            # Cut off at EOS if present.
            if eos_token_id is None:
                eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id in gen_tokens:
                idx = gen_tokens.index(eos_token_id)
                gen_tokens = gen_tokens[:idx]
            outputs.append(self.tokenizer.decode(gen_tokens))
            print(f"Prompt: {self.tokenizer.decode(prompt_tokens[i])}\n 'Generated':, {self.tokenizer.decode(gen_tokens)}")
        return outputs





