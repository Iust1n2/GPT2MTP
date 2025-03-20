"""
Utilities for attention components and MTP loss. Some are taken from transformer_lens library
"""
import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from typing import Optional, Union

# both taken from transformer_lens.utilities.attention
def simple_attn_linear(
    input: Float[torch.Tensor, "batch pos d_model"],
    w: Float[torch.Tensor, "head_index d_model d_head"],
    b: Float[torch.Tensor, "head_index d_head"],
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    """Linear layer for attention calculation."""
    w = einops.rearrange(w, "head_index d_model d_head -> (head_index d_head) d_model")
    b_ = einops.rearrange(b, "head_index d_head -> (head_index d_head)")
    return F.linear(input, w, b_).reshape(input.shape[0], input.shape[1], b.shape[0], b.shape[1])


def complex_attn_linear(
    input: Float[torch.Tensor, "batch pos head_index d_model"],
    w: Float[torch.Tensor, "head_index d_model d_head"],
    b: Float[torch.Tensor, "head_index d_head"],
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    """Linear layer for attention calculation.

    This is almost the same as simple_attn_linear, but the input tensor has an extra head_index dimension, used when calculating the input of each attention head separately.
    """
    return (
        einops.einsum(
            input,
            w,
            "batch pos head_index d_model, head_index d_model d_head -> batch pos head_index d_head",
        )
        + b
    )

def calculate_attn_scores(
    q_input: Float[torch.Tensor, "batch pos head_index d_head"],
    k_input: Float[torch.Tensor, "batch pos head_index d_head"],
    attn_scale: float,
) -> Float[torch.Tensor, "batch head_index query_pos key_pos"]:
    """Calculate attention scores."""
    q_ = einops.rearrange(
            q_input, "batch query_pos head_index d_head -> batch head_index query_pos d_head"
            )
    k_ = einops.rearrange(
                k_input, "batch key_pos head_index d_head -> batch head_index d_head key_pos"
            )

    # calculate the attention scores
    attn_scores = q_ @ k_ / attn_scale # torch.Size([1, 32, 18, 18])
    return attn_scores

def apply_causal_mask(
    attn_scores: Float[torch.Tensor, "batch head_index pos pos_plus_past_kv_pos_offset"],
    past_kv_pos_offset: int = 0,
    attention_mask: Optional[Int[torch.Tensor, "batch offset_pos"]] = None,
    mask: Optional[Int[torch.Tensor, "pos pos_plus_past_kv_pos_offset"]] = None,
) -> Float[torch.Tensor, "batch head_index pos pos_plus_past_kv_pos_offset"]:
        # The query context length is the number of positions we take queries from - if not using a past_kv_cache this is just the context length (for the current prompt), but if we're caching it can be different.
        query_ctx_length = attn_scores.size(-2)
        # The key context length is the number of positions in the past - this includes all positions in the cache
        # If not caching, query_ctx_length == key_ctx_length
        key_ctx_length = attn_scores.size(-1)

        if query_ctx_length + past_kv_pos_offset != key_ctx_length:
            raise ValueError(
                f"query_ctx_length {query_ctx_length} + past_kv_pos_offset {past_kv_pos_offset} != key_ctx_length {key_ctx_length} - you likely have a bug."
            )

        # Index back to front to ensure local attention works
        final_mask = mask[None, None, -query_ctx_length:, -key_ctx_length:]  # [1, 1, pos, pos]
        if attention_mask is not None:
            # Apply a causal mask to the attention scores considering the padding
            einsum_str = "batch head pos offset_pos, batch offset_pos -> batch head pos offset_pos"
            final_mask = final_mask.to(attention_mask.device)
            final_mask = einops.einsum(final_mask, attention_mask, einsum_str).bool()
        attn_scores = attn_scores.to(final_mask.device) 
        ignore_val = torch.tensor(-1e4, dtype=attn_scores.dtype, device=attn_scores.device)
        return torch.where(final_mask, attn_scores, ignore_val)


def lm_cross_entropy_loss(
    logits: Float[torch.Tensor, "batch pos d_vocab"],
    targets: Int[torch.Tensor, "batch pos"],
    attention_mask: Optional[Int[torch.Tensor, "batch pos"]] = None,
    per_token: bool = False,
    pad_token_id: Optional[int] = None,
) -> Union[Float[torch.Tensor, ""], Float[torch.Tensor, "batch pos"]]:
    """Cross entropy loss for the language model, gives the loss for predicting the NEXT token.

    Args:
        logits (torch.Tensor): Logits. Shape [batch, pos, d_vocab]
        tokens (torch.Tensor[int64]): Input tokens. Shape [batch, pos]
        attention_mask (torch.Tensor[int64], optional): Attention mask. Shape [batch, pos]. Used to
            mask out padding tokens. Defaults to None.
        per_token (bool, optional): Whether to return the log probs predicted for the correct token, or the loss (ie mean of the predicted log probs). Note that the returned array has shape [batch, seq-1] as we cannot predict the first token (alternately, we ignore the final logit). Defaults to False.
        pad_token_id (int, optional): Padding token id to ingore in the loss computation. Defaults to None.
    """        
    if logits.dim() == 4 and logits.size(2) == 1:
        logits = logits.squeeze(2)  # Now shape: [batch, pos, d_vocab]

    # Extract the logits for the next token
    logits_last = logits[:, :-1, :]  # shape: [batch, pos-1, d_vocab]

    # Flatten for cross_entropy: logits_i becomes (batch*pos, d_vocab) and targets_i becomes (batch*pos)
    logits_flat = logits_last.reshape(-1, logits_last.size(-1))
    targets_flat = targets.reshape(-1)

    # Compute per-token cross-entropy loss.
    loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=pad_token_id)  # shape: (batch*pos)
        
    # If an attention mask is provided, apply it.
    if attention_mask is not None:
        loss = loss * attention_mask  # zeros out loss where attention_mask==0
        if not per_token:
            # Average only over non-masked tokens.
            loss = loss.sum() / attention_mask.sum()
            return loss
        else:
            return loss
    else:
        if not per_token:
            return loss.mean()
        else:
            return loss


def mtp_cross_entropy_loss(
    logits: torch.Tensor,    # shape: (batch, pos, n_future, d_vocab)
    targets: torch.Tensor,   # shape: (batch, pos + n_future)
    attention_mask: Optional[torch.Tensor] = None,  # shape: (batch, pos + n_future)
    per_token: bool = False,
    pad_token_id: int = None,
) -> Union[torch.Tensor, torch.Tensor]:
    """
    Multi-token prediction cross-entropy loss for targets that are shifted by n_future positions.
    
    The model produces logits of shape (batch, pos, n_future, d_vocab) where the i-th head predicts the token 
    at position t+i+1 for context position t (with t in [0, pos)). The targets tensor has shape 
    (batch, pos + n_future), with the first pos tokens serving as context and the remaining tokens as targets.
    
    For head i, predictions (from logits[:, :pos, i, :]) are compared with targets[:, i+1 : i+1+pos].
    
    Args:
        logits (torch.Tensor): Predicted logits with shape (batch, pos, n_future, d_vocab).
        targets (torch.Tensor): Ground truth token ids with shape (batch, pos + n_future).
        attention_mask (torch.Tensor, optional): Mask of shape (batch, pos + n_future) where positions with 0 (or False)
            are ignored.
        per_token (bool, optional): If True, return per-token losses as a tensor; otherwise, return the average loss.
        pad_token_id (int, optional): EOT token to ignore in the loss computation.
    Returns:
        Either a scalar loss (if per_token is False) or a tensor of per-token losses.
    """
    # Compute log probabilities over the vocabulary (if needed for some debugging, but we compute loss directly below)
    # Here we use logits directly with F.cross_entropy.
    batch, pos, n_future, d_vocab = logits.shape

    logits_last = logits[:, pos - 1, :, :]

    loss_sum = 0.0
    token_count = 0
    per_token_losses = []  # will hold per-token losses if requested

    # For each future head, head i predicts token at position t+i+1 for t in [0, pos)
    for i in range(n_future):
        # Here the valid positions for the context are simply the full pos, because
        # targets has shape (batch, pos + n_future)
    
        # Extract the logits for head i: shape (batch, pos, d_vocab)
        logits_i = logits_last[:, i, :]
        # Extract the corresponding targets:
        # For context position t, the target is at index t+i+1 in the targets tensor.
        targets_i = targets[:, i]        
        # Flatten for cross_entropy: logits_i becomes (batch*pos, d_vocab) and targets_i becomes (batch*pos)
        logits_i_flat = logits_i.reshape(-1, d_vocab)
        targets_i_flat = targets_i.reshape(-1)

        # Compute per-token cross-entropy loss with no reduction.
        loss_i = F.cross_entropy(logits_i_flat, targets_i_flat, ignore_index=pad_token_id)  # shape: (batch*pos)
        
        if attention_mask is not None:
            # Construct masks for context and target positions.
            mask_context = attention_mask[:, :pos].bool()  # shape: (batch, pos)
            mask_target  = attention_mask[:, i+1 : i+1+pos].bool()  # shape: (batch, pos)
            valid_mask   = mask_context & mask_target  # shape: (batch, pos)
            valid_mask_flat = valid_mask.reshape(-1).float()
            loss_i = loss_i * valid_mask_flat
            n_tokens = valid_mask_flat.sum().item()
        else:
            n_tokens = targets_i_flat.numel()
        
        loss_sum += loss_i.sum()
        token_count += n_tokens

        if per_token:
            # Save per-token losses reshaped back to (batch, pos)
            per_token_losses.append(loss_i.view(batch, pos))
    
    if per_token:
        # Concatenate along the sequence dimension: result has shape (batch, n_future * pos)
        return torch.cat(per_token_losses, dim=1)
    else:
        return loss_sum / token_count
    
def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
