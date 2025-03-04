import os
from dataclasses import dataclass
from pathlib import Path
import logging
import pickle
import time
from tqdm import tqdm
from omegaconf import OmegaConf
from contextlib import nullcontext

import math
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.profiler import profile, record_function, ProfilerActivity

from data.dataset import OpenWebTextDataset, DatasetConfig
from model.model_base import GPT2MTP, GPT2MTPConfig
from transformers import get_cosine_schedule_with_warmup

logging.getLogger().setLevel(logging.INFO)

@dataclass
class TrainingArgs: 
    model_config : GPT2MTPConfig
    dataset_config : DatasetConfig
    data_dir : str = 'data/open_web_text_10k/'  
    save_dir : str = 'checkpoints/open_web_text_10k/'
    init_from : str = 'scratch' # or 'resume'
    always_save_checkpoint : bool = True
    eval_only : bool = False 
    eval_interval : int = 500
    eval_iters : int = 200
    n_future : int = 4
    batch_size : int = 32 # gpt2 uses batch size of 512 for 1024 tokens in context
    gradient_accumulation_steps : int = 3 * 32
    max_iters : int = 10850 
    lr_decay_iters : int = 10850
    max_lr : float = 3e-4 
    decay_lr : bool = True
    warmup_steps : int = 1000
    min_lr : float = 1e-5
    weight_decay : float = 0.1
    betas : tuple = (0.9, 0.95)
    grad_clip : float = 1.0
    log_interval : int = 1
    device = 'cuda' 
    dtype : str = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile : bool = True
    
def compute_ngram_accuracy(ngram, pred_tokens, target_tokens, ngrams_cond):
    correct, total = 0, 0
    for pred_seq, target_seq in zip(pred_tokens, target_tokens):
        seq_len = len(pred_seq)

        for i in range(seq_len - ngram):
            pred_ngram = tuple(pred_seq[i:i+ngram].tolist())   
            target_ngram = tuple(target_seq[i:i+ngram].tolist())
        
            # Only count if target n-gram exists in `ngrams_cond`
            if ngrams_cond.get(target_ngram[:-1], {}).get(target_ngram[-1], 0) > 0:
                total += 1
                if pred_ngram == target_ngram:
                    correct += 1

    return correct, total

def save_checkpoint(save_dir, model, optimizer, scheduler, step, best_val_loss, history):
    """Save model checkpoint."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': step,
        'best_val_loss': best_val_loss,
        'history': history,
        # 'config': model.config
    }, os.path.join(save_dir, f'checkpoint_step_{step}.pt'))

if __name__ == "__main__":
    args = TrainingArgs(
        model_config = GPT2MTPConfig(),
        dataset_config = DatasetConfig(data_dir='data/open_web_text_10k/', split='train'),
    )
    cfg = OmegaConf.merge(OmegaConf.structured(args), OmegaConf.from_cli())
    
    if args.save_dir is not None:
        outdir = Path('') / Path(args.save_dir)
        outdir.mkdir(parents=True, exist_ok=True)

    # Load training metadata
    with open(os.path.join(args.data_dir, 'train_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    bigrams_cond = meta['bigrams_cond']
    trigrams_cond = meta['trigrams_cond']
    fourgrams_cond = meta['fourgrams_cond']

    # Load dataset
    train_dataset = OpenWebTextDataset(args=args.dataset_config, split='train')

    # Config of I/O precision
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    torch._dynamo.config.suppress_errors = True
    # torch.set_float32_matmul_precision('medium')  # allow tf32 on matmul
    # torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in args.device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else autocast(device_type=device_type, dtype=ptdtype)

    # Profiling exec time
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    sort_by_keyword = args.device + "_time_total"
    profile_dir = f'profiling/batch_size_{args.batch_size}_block_size_{args.dataset_config.block_size}'
    os.makedirs(profile_dir, exist_ok=True)

    tokens_per_step = args.gradient_accumulation_steps * args.batch_size * args.dataset_config.block_size
    print(f"tokens per step will be: {tokens_per_step:,}")

    step = 0
    best_val_loss = 1e9
    if args.init_from == 'scratch':
        # Initialize model and init weights
        model = GPT2MTP(config=args.model_config, n_future=args.n_future).to(args.model_config.device) 
        model.init_weights()
        model.train()
    elif args.init_from == 'resume':
        print(f"Resuming training from {args.save_dir}")
        ckpt_path = os.path.join(args.save_dir, 'checkpoint_step.pt')
        ckpt = torch.load(ckpt_path, map_location=args.model_config.device)
        ckpt_model_args = ckpt['model_args']
        model = GPT2MTP(config=ckpt_model_args, n_future=args.n_future).to(args.model_config.device)
        state_dict = ckpt['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        step = ckpt['step']
        best_val_loss = ckpt['best_val_loss']
    
    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=args.max_lr, betas=args.betas, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_iters)
    scaler = GradScaler('cuda')
    
    if args.init_from == 'resume':
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
    
    if args.compile: 
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)

    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_steps:
            return args.max_lr * (it + 1) / (args.warmup_steps + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        if it > args.lr_decay_iters:
            return args.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - args.warmup_steps) / (args.lr_decay_iters - args.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return args.min_lr + coeff * (args.max_lr - args.min_lr)
    
    # Helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                    X, Y = train_dataset.get_batch(split)
                    with ctx:
                        loss = model(X, Y, return_type='loss')  
                    losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
    history, token_history = [], []
    t0 = time.time()
    
    # Training loop
    while step < args.max_iters:
        lr = get_lr(step) if args.decay_lr else args.max_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
        # Evaluate the loss on train/val sets and write checkpoints
        if step % args.eval_interval == 0:
            losses = estimate_loss()
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses['val'] < best_val_loss or args.always_save_checkpoint:
                best_val_loss = losses['val']
                save_checkpoint(args.save_dir, model, optimizer, scheduler, step, best_val_loss, history)
        
        # Do only an eval step
        if step == 0 and args.eval_only:
            break
        
        all_hook_names = list(model.hook_dict.keys())
        cache = model.add_caching_hooks(names_filter=all_hook_names, incl_bwd=True) 
        
        # Gradient accumulation loop
        pbar = tqdm(range(args.gradient_accumulation_steps), desc=f"Step {step}/{args.max_iters}")
        for micro_step in pbar:
            print(f"Micro-step {micro_step + 1}/{args.gradient_accumulation_steps}")
            total_loss = 0.0
            # 0. Get batch
            X, y = train_dataset.get_batch('train')
            with ctx:
                # (Optional) Save token history for logging:
                token_history.extend(X.flatten().tolist())

                prompt_len = X.shape[1] - model.n_future  # How much of X is prompt
                target_len = model.n_future  # How many tokens to predict           

                with profile(activities=activities, profile_memory=True, record_shapes=True) as prof1:
                    with record_function("fwd_shared_trunk"):
                        # 1. Run the shared trunk once.
                        z = model.shared_trunk(X)  # shape: (batch, seq, d_model)
                # print(prof1.key_averages().table(sort_by=sort_by_keyword, row_limit=10))
                prof1.export_chrome_trace(f"{profile_dir}/trace_fwd_trunk.json")
                    # 2. Detach z and set requires_grad so gradients from heads are collected.
                d = z.detach().clone().requires_grad_()
                
                # # (Optional) Full fwd to ensure that *all* hooks are triggered
                # logits = model(X, y, return_type="logits")  # shape (batch, seq_length, n_future vocab_size)
                # final_logits = logits[:, :-1, :, :]  # Shape: (batch, final_pos, n_future, vocab_size)
                
                # 3. For each MTP head, compute the output and loss.
                for i in range(model.n_future):
                    # Forward pass through the MTP head to get the projected residual stream vector and hook it
                    with profile(activities=activities, profile_memory=True, record_shapes=True) as prof2:
                        with record_function("fwd_mtp_head"):
                            mtp_resid_post = model.hook_mtp_heads[i](model.mtp_heads[i](d))    # (batch, seq_length, d_model)
                    # print(prof2.key_averages().table(sort_by=sort_by_keyword, row_limit=10))
                    prof2.export_chrome_trace(f"{profile_dir}/trace_fwd_mtp.json")
                
                    # Unembed the output of the MTP head
                    head_logits = model.unembed(mtp_resid_post)   # (batch, seq_length, vocab_size)

                    # Calculate valid length based on head offset (prevent index overflow)
                    valid_length = head_logits.size(1) - (i + 1)
                    if valid_length <= 0:
                        continue  # Skip if no valid target positions are available

                    # Align logits and targets
                    # head_logits = logits[:, :, i, :]
                    logits_i = head_logits[:, prompt_len-1 : prompt_len-1+target_len, :]
                    targets_i = y[:, prompt_len : prompt_len+target_len]

                    # Check target range
                    vocab_size = logits_i.size(-1)
                    assert targets_i.min() >= 0, f"Invalid target: min={targets_i.min()}"
                    assert targets_i.max() < vocab_size, f"Invalid target: max={targets_i.max()} ≥ vocab_size={vocab_size}"

                    # 4. Compute cross-entropy loss per head
                    loss_i = F.cross_entropy(
                        logits_i.reshape(-1, logits_i.size(-1)),  # (batch * valid_length, vocab_size)
                        targets_i.reshape(-1),                    # (batch * valid_length)
                        # ignore_index=-1                           # Ignore padding if used
                    )
            
                    total_loss += loss_i
                    # 5. Backward for this head.
                    # Retain the computational graph if more heads will be processed.
                    scaler.scale(loss_i).backward(retain_graph=(i < args.n_future - 1))

                    # Print the gradient norm for this head
                    if d.grad is not None:
                        grad_norm = d.grad.norm().item()
                        print(f"Gradient norm for MTP head {i + 1}: {grad_norm:.4f}, Head Loss: {loss_i.item():.4f}")
                    else:
                        print(f"Warning: d.grad is None after backward pass of head {i + 1}")

                    # Free memory
                    del head_logits
                    torch.cuda.empty_cache()
            
            # # Save hooks but don't use in computation (won't work)
            # mtp_outputs_pre = model.hook_mtp_heads_out_pre(mtp_resid_post)
            # mtp_outputs_post = model.hook_mtp_heads_out_post(model.ln_f(mtp_resid_post))

            # TODO: fix hook_mtp_heads_out_pre/post and unembed not being saved in cache 
            # due to not actually running a full forward pass model(X) inside the optimization loop
            # the forward pass stops before the add_mtp_heads hook
            if "hook_mtp_heads_out_pre_grad" in cache:
                # print(f"Grad Norm for hook_mtp_heads_out_post: {cache['hook_mtp_heads_out_post'].norm().item():.4f}")            
                print(f"Activation shape: {cache['hook_mtp_heads_out_pre_grad'].shape}")
            if "hook_mtp_heads_out_post_grad" in cache:
                # print(f"Grad Norm for hook_mtp_heads_out_post: {cache['hook_mtp_heads_out_post'].norm().item():.4f}")            
                print(f"Activation shape: {cache['hook_mtp_heads_out_post_grad'].shape}")
            if "hook_unembed_grad" in cache:
                # print(f"Grad Norm for hook_mtp_heads_out_post: {cache['hook_mtp_heads_out_post'].norm().item():.4f}")            
                print(f"Activation shape: {cache['hook_unembed_grad'].shape}")

            # prefetch the next batch and define the target n-grams
            X, Y = train_dataset.get_batch('train')
            y_bigram = y[:, :1]  
            y_trigram = y[:, :2]  
            y_fourgram = y[:, :3]
            
            # 6. Propagate accumulated gradients through the trunk
            if d.grad is not None:
                z.backward(d.grad)
                logging.info(f"Final gradient norm after all heads: {d.grad.norm().item():.4f}, Total Loss: {total_loss.mean().item():.4f}, Average Loss: {(total_loss / args.n_future).item():.4f}")
            else:
                raise ValueError("d.grad is None. Ensure loss.backward() was called.")
        
        # 7. Gradient clipping
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), args.grad_clip)
    
        # 8. Optimizer step 
        total_loss = total_loss / args.gradient_accumulation_steps # scale the loss to account for gradient accumulation
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        # Log final loss
        if step % args.log_interval == 0:
            loss_value = total_loss.item() * args.gradient_accumulation_steps  # unscale loss for logging
            logging.info(f"Step {step}: loss {loss_value:.2f}, time {dt:.2f}s") 
        
        # Compute n-gram accuracies
        with torch.no_grad():
            # get the logits on the final position
            logits = model.forward(X)[:, -1, :, :]  #  shape: (batch, n_future, vocab_size)
            pred_token_ids = torch.argmax(logits, dim=-1)  # shape: (total_batch, target_len)
            predicted_future_toks = model.to_string(pred_token_ids)
            target_future_toks = model.to_string(y[:, :args.n_future])

        # Extract n-grams
        pred_bigrams = pred_token_ids[:, :1] 
        pred_trigrams = pred_token_ids[:, :2] 
        pred_fourgrams = pred_token_ids[:, :3] 

        # # possible combination of ngrams
        # pred_bigrams = [tuple(pred_token_ids[:, i : i+1].tolist()) for i in range(pred_token_ids.shape[1] - 1)]
        # pred_trigrams = [tuple(pred_token_ids[:, i : i+2].tolist()) for i in range(pred_token_ids.shape[1] - 2)]
        # pred_fourgrams = [tuple(pred_token_ids[:, i : i+3].tolist()) for i in range(pred_token_ids.shape[1] - 3)]

        for predicted_tokens, target_tokens in zip(predicted_future_toks, target_future_toks):
            logging.info(f"Predicted 4gram: {predicted_tokens}, Target 4gram: {target_tokens}")
        # Compute n-gram accuracies using your provided functions.
        correct_2, total_2 = compute_ngram_accuracy(2, pred_bigrams, y_bigram, bigrams_cond)
        correct_3, total_3 = compute_ngram_accuracy(3, pred_trigrams, y_trigram, trigrams_cond)
        correct_4, total_4 = compute_ngram_accuracy(4, pred_fourgrams, y_fourgram, fourgrams_cond)
        accuracy_2 = correct_2 / total_2 if total_2 > 0 else 0
        accuracy_3 = correct_3 / total_3 if total_3 > 0 else 0
        accuracy_4 = correct_4 / total_4 if total_4 > 0 else 0

        # Log statistics.
        curr_history = {
            'step': step,
            'loss': loss_value,
            'bigram_accuracy': accuracy_2,
            'trigram_accuracy': accuracy_3,
            'fourgram_accuracy': accuracy_4
        }
        history.append(curr_history)
        logging.info(f"Step {step}: 2-gram accuracy: {accuracy_2:.2f}, 3-gram accuracy: {accuracy_3:.2f}, 4-gram accuracy: {accuracy_4:.2f}")
        
        # Save checkpoint every 500 steps.
        if step % 500 == 0 and step > 0:
            save_checkpoint(model, optimizer, scheduler, step, args.save_dir, history)
        
        step += 1
    save_checkpoint(model, optimizer, scheduler, step, args.save_dir, history)
    logging.info(f"Training complete in {dt:.2f} seconds.") 