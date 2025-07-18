import os
from dataclasses import dataclass
from pathlib import Path
import logging
import pickle
import time
from contextlib import nullcontext

import gc
import re
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.profiler import profile, record_function, ProfilerActivity

from data.dataset import OpenWebTextDataset, DatasetConfig
from model.model_base import GPT2MTP, GPT2MTPConfig
from transformers import get_cosine_schedule_with_warmup
from train_utils import (
    save_checkpoint,
    plot_grad_flow, 
    log_acts_and_grads,
    compute_ngram_accuracy
)

logging.getLogger().setLevel(logging.INFO)

@dataclass
class TrainingArgs: 
    model_config : GPT2MTPConfig
    dataset_config : DatasetConfig
    data_dir : str = 'data/open_web_text/'  
    save_dir : str = 'checkpoints/open_web_text/'
    init_from : str = 'resume' # or 'resume'
    always_save_checkpoint : bool = True
    eval_only : bool = False 
    eval_interval : int = 400
    eval_iters : int = 200
    eval_ngrams : bool = False
    n_future : int = 4
    batch_size : int = 16 # gpt2 uses batch size of 512 for 1024 tokens in context
    gradient_accumulation_steps : int = 128 # 2**19 tokens per step / (bsz=16 * block_size=256) = 128
    max_iters : int = 10850 
    lr_decay_iters : int = 10850
    max_lr : float = 3e-4 
    decay_lr : bool = True
    warmup_steps : int = 1000
    min_lr : float = 3e-5
    weight_decay : float = 0.1 # reduce from 0.03 to 0.01 if residual stream gradients begin to vanish in deeper layers or overall smaller than mtp heads
    betas : tuple = (0.9, 0.95)
    grad_clip : float = 1.0 # increase to 1.5 if mtp heads gradients start exploding
    log_interval : int = 200
    device : str = 'cuda' 
    dtype : str = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile : bool = True

if __name__ == "__main__":
    os.environ['TOKENIZERS_PARALLELISM']='false'
    args = TrainingArgs(
        model_config = GPT2MTPConfig(),
        dataset_config = DatasetConfig(data_dir='data/open_web_text/', split='train'),
    )
    logging.info(f"Training with args: {args}")

    if args.save_dir is not None:
        outdir = Path('') / Path(args.save_dir)
        outdir.mkdir(parents=True, exist_ok=True)

    if args.eval_ngrams: 
        # Load n-gram conditionals
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
    torch._dynamo.config.cache_size_limit = 128  # Increase as needed
    torch.set_float32_matmul_precision('high')  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
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
    logging.info(f"Tokens per step will be: {tokens_per_step:,}")

    step = 0
    best_val_loss = 1e9
    if args.init_from == 'scratch':
        # Initialize model and init weights
        model = GPT2MTP(config=args.model_config, n_future=args.n_future).to(args.model_config.device) 
        model.init_weights()
    elif args.init_from == 'resume':
        ckpt_path = os.path.join(args.save_dir, 'checkpoint_step_800.pt') # Load a preferred checkpoint
        logging.info(f"Resuming training from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=args.model_config.device)
        model = GPT2MTP(config=args.model_config, n_future=args.n_future).to(args.model_config.device)
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
    scaler = GradScaler(enabled=(args.dtype == 'float16'))
    
    if args.init_from == 'resume':
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Resumed Training - Step {step}, Learning Rate: {current_lr:.8f}")
    
    if args.compile: 
        logging.info("Compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)
        logging.info("Compilation complete.")

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
    
    if args.init_from == 'resume':
        history = ckpt['history']
        tokens_seen = history[-1]['tokens_seen']
        if isinstance(tokens_seen, list):
            token_history = tokens_seen
        else:
            token_history.append(tokens_seen)
        
    # Get the first batch
    X, y = train_dataset.get_batch('train') 

    # Training loop
    while step < args.max_iters:
        # Get the current learning rate (either restored or new)
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Step {step}, Learning Rate: {current_lr}")
    
        # Add hooks for caching forward activations 
        all_hook_names = list(model.hook_dict.keys())
        cache = model.add_caching_hooks(names_filter=all_hook_names, incl_bwd=True) 
        grads_for_vis = [
            re.compile(r'^(hook_mtp_heads)\.(\d+)_grad$'),
            re.compile(r'^(blocks\.(\d+)\.hook_resid_post)_grad$')
        ]
        # Evaluate the loss on train/val sets and write checkpoints
        if step % args.eval_interval == 0:
            losses = estimate_loss()
            logging.info(f"Eval Step {step}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")
            if losses['val'] < best_val_loss or args.always_save_checkpoint:
                best_val_loss = losses['val']
                save_checkpoint(args.save_dir, model, optimizer, scheduler, step, best_val_loss, history, cache)
        
        # Do only an eval step
        if step == 0 and args.eval_only:
            break
        
        # Gradient accumulation loop 16 (batch_size) * (64) (gradient_accumulation_steps) = 1024 effective batch_size per optimizer step 
        # tokens per effective batch = 524,288
        total_loss_accum = 0.0
        logging.info(f"Entering gradient accumulation loop for step {step}")
        for micro_step in range(args.gradient_accumulation_steps):
            # print(f"Micro-step {micro_step + 1}/{args.gradient_accumulation_steps}")
            micro_loss = 0.0
            head_losses = {}
    
            # Save token history:
            token_history.extend(X.flatten().tolist())
            
            # We subtract n_future from the sequence length to match the causal MTP setting 
            prompt_len = X.shape[1] - model.n_future 

            # 2. Run the forward pass on the shared trunk once with autocast
            with ctx:
                z = model.shared_trunk(X)  # shape: (batch, seq, d_model)
    
            # 3. Detach z and set requires_grad so gradients from heads are collected.
            d = z.detach().clone()
            d.requires_grad = True

            # 4. For each MTP head, compute the output and loss and accumulate their gradients wrt to *their* loss in d.grad of the shared trunk z
            # for i in range(model.n_future): # anti-causal
            for i in reversed(range(model.n_future)): # causal
                with ctx:
                    # 3. # Run the forward pass through the furthest MTP head to get the projected residual stream vector and hook it
                    mtp_resid_post = model.hook_mtp_heads[i](model.mtp_heads[i](d))    # (batch, seq_length, d_model)
                
                    # 4. Unembed the output of the MTP head
                    head_logits = model.unembed(mtp_resid_post)   # (batch, seq_length, vocab_size)

                    # Align logits and targets for causal prediction
                    logits_i = head_logits[:, prompt_len - 1 + i, :]  # Predict token at (prompt_len + i)
                    targets_i = y[:, prompt_len + i]

                    # 5. Compute cross-entropy loss per head
                    loss_i = F.cross_entropy(
                        logits_i.reshape(-1, logits_i.size(-1)),  # (batch * valid_length, vocab_size)
                        targets_i.reshape(-1),                    # (batch * valid_length)
                        # ignore_index=-1                           # Ignore padding if used
                    )
                # End autocast before the backward pass
                # scale the loss to account for gradient accumulation
                loss_i_scaled = loss_i / args.gradient_accumulation_steps
                micro_loss += loss_i_scaled.detach()
                head_losses[f'step_{step}_micro_step_{micro_step}_head_{i}'] = loss_i_scaled.item()
                
                # 6. Backward for this head.
                # Retain the computational graph if more heads will be processed.
                # scaler.scale(loss_i_scaled).backward(retain_graph=(i < args.n_future - 1)) # anti-causal
                scaler.scale(loss_i_scaled).backward(retain_graph=(i > 0)) # causal

                # Print the gradient norm for this head
                if d.grad is not None:
                    grad_norm = d.grad.norm().item()
                    if micro_step % args.gradient_accumulation_steps == 0:
                        logging.info(f"Final Micro-step {micro_step + args.gradient_accumulation_steps}, Gradient norm for MTP head {i + 1}: {grad_norm:.4f}, Head Loss: {loss_i_scaled.item():.4f}")
                else:
                    print(f"Warning: d.grad is None after backward pass of head {i + 1}")
                # Free memory
                del mtp_resid_post, head_logits, logits_i, loss_i, loss_i_scaled
                gc.collect()
                torch.cuda.empty_cache()

            # Accumulate the total loss
            total_loss_accum += micro_loss
        # prefetch the next batch and define the target n-grams
        X, y = train_dataset.get_batch('train')

        # 7. Propagate accumulated gradients through the trunk
        if d.grad is not None:
            z.backward(d.grad)
            logging.info(f"Final accum gradient norm after all heads: {d.grad.norm().item():.4f}, Total Loss: {total_loss_accum.item():.4f}, Average Loss: {((total_loss_accum / args.n_future)).item():.4f}")
        else:
            raise ValueError("d.grad is None. Ensure loss.backward() was called.")
                
        # Log gradients before clipping
        if step % args.log_interval == 0:
            plot_grad_flow(cache, grads_for_vis, step)
            log_acts_and_grads(cache, step)
        
        # 8. Gradient clipping
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            norm = clip_grad_norm_(model.parameters(), args.grad_clip) # norm = accumulated gradient norm *before* clipping

        # 9. Optimizer step  
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        average_loss_per_token = total_loss_accum / model.n_future
        # Log final loss
        if step % args.log_interval == 0:
            # unscale loss for logging
            # loss_value = total_loss.item() * args.gradient_accumulation_steps / model.n_future
            logging.info(f"Step {step}: Loss {average_loss_per_token:.2f}, Norm: {norm:.4f}, Time: {dt:.2f}s, Tokens Seen: {len(token_history)/ 1e6:.1f}M") 
        
        curr_history = {
                'step': step,
                'loss': average_loss_per_token.item(),
                'lr': current_lr,
                'head_losses': head_losses,
                'tokens_seen': len(token_history),
            }
        if step % args.eval_iters == 0:
            y_bigram = y[:, :1]  
            y_trigram = y[:, :2]  
            y_fourgram = y[:, :3]
                
            # Compute n-gram accuracies
            with torch.no_grad():
                logits = model.forward(X)[:, -1, :, :]  # (batch, n_future, vocab_size)
                pred_token_ids = torch.argmax(logits, dim=-1)  # (batch, n_future)
                predicted_future_toks = [model.tokenizer.batch_decode(tok_seq) for tok_seq in pred_token_ids]
                target_future_toks = [model.tokenizer.batch_decode(tgt_seq[:args.n_future]) for tgt_seq in y]

            for predicted_tokens, target_tokens in zip(predicted_future_toks, target_future_toks):
                logging.info(f"Predicted n-gram: {predicted_tokens}, Target n-gram: {target_tokens}")

            # Extract n-grams
            pred_bigrams = pred_token_ids[:, :1] 
            pred_trigrams = pred_token_ids[:, :2] 
            pred_fourgrams = pred_token_ids[:, :3] 

            # # possible combination of ngrams
            # pred_bigrams = [tuple(pred_token_ids[:, i : i+1].tolist()) for i in range(pred_token_ids.shape[1] - 1)]
            # pred_trigrams = [tuple(pred_token_ids[:, i : i+2].tolist()) for i in range(pred_token_ids.shape[1] - 2)]
            # pred_fourgrams = [tuple(pred_token_ids[:, i : i+3].tolist()) for i in range(pred_token_ids.shape[1] - 3)]

            if args.eval_ngrams and bigrams_cond is not None:
                # Compute n-gram accuracies using your provided functions.
                correct_2, total_2 = compute_ngram_accuracy(2, pred_bigrams, y_bigram, bigrams_cond)
                correct_3, total_3 = compute_ngram_accuracy(3, pred_trigrams, y_trigram, trigrams_cond)
                correct_4, total_4 = compute_ngram_accuracy(4, pred_fourgrams, y_fourgram, fourgrams_cond)
                accuracy_2 = correct_2 / total_2 if total_2 > 0 else 0
                accuracy_3 = correct_3 / total_3 if total_3 > 0 else 0
                accuracy_4 = correct_4 / total_4 if total_4 > 0 else 0

                # Log statistics.
                curr_history['bigram_accuracy'] = accuracy_2
                curr_history['trigram_accuracy'] = accuracy_3
                curr_history['fourgram_accuracy'] = accuracy_4
                
                logging.info(f"Step {step}: 2-gram accuracy: {accuracy_2:.2f}, 3-gram accuracy: {accuracy_3:.2f}, 4-gram accuracy: {accuracy_4:.2f}")
        
        history.append(curr_history)
    
        step += 1
    # Save final checkpoint
    save_checkpoint(model, optimizer, scheduler, step, args.save_dir, history, cache)
    logging.info(f"Training complete in {dt:.2f} seconds.") 