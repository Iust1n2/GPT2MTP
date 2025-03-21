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
from torch.profiler import ProfilerActivity

from data.dataset import OpenWebTextDataset, DatasetConfig
from model.model_base import GPT2MTP, GPT2MTPConfig
from transformers import get_cosine_schedule_with_warmup
from train.utils import (
    save_checkpoint,
    plot_grad_flow, 
    log_acts_and_grads,
    compute_ngram_accuracy,
    generate_text
)

logging.getLogger().setLevel(logging.INFO)

@dataclass
class TrainingArgs: 
    model_config : GPT2MTPConfig
    dataset_config : DatasetConfig
    # data_dir : str = 'data/delphi_stories/'  
    save_dir : str = 'checkpoints/delphi_stories/'
    init_from : str = 'scratch' # or 'resume'
    always_save_checkpoint : bool = True
    eval_only : bool = False 
    eval_interval : int = 400 // 4
    eval_iters : int = 200 // 4
    eval_ngrams : bool = False
    n_future : int = 4
    batch_size : int = 16 # gpt2 uses batch size of 512 for 1024 tokens in context
    gradient_accumulation_steps : int = 64 # 2**19 tokens per step / (bsz=16 * block_size=256) = 128
    max_iters : int = 2080 
    lr_decay_iters : int = 2080
    max_lr : float = 3e-4 
    decay_lr : bool = True
    warmup_steps : int = 1000 // 20
    min_lr : float = 3e-5
    weight_decay : float = 0.1 # reduce from 0.03 to 0.01 if residual stream gradients begin to vanish in deeper layers or overall smaller than mtp heads
    betas : tuple = (0.9, 0.95)
    grad_clip : float = 1.0 # increase to 1.5 if mtp heads gradients start exploding
    log_interval : int = 200 // 4
    device : str = 'cuda' 
    dtype : str = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile : bool = True

if __name__ == "__main__":
    os.environ['TOKENIZERS_PARALLELISM']='false'
    args = TrainingArgs(
        model_config = GPT2MTPConfig(n_ctx=512, n_layers=2, n_heads=4, d_model=256, d_mlp=1024, d_head=64, tokenizer_name='delphi-suite/stories-tokenizer'),
        dataset_config = DatasetConfig(data_dir='data/delphi_stories/', split='train', batch_size=16, block_size=512),
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
    train_dataset = OpenWebTextDataset(args.dataset_config, split='train')

    # Config of I/O precision
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    # torch._dynamo.config.suppress_errors = True
    # torch._dynamo.config.cache_size_limit = 128  # Increase as needed
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
        ckpt_path = os.path.join(args.save_dir, 'checkpoint_step_1000.pt') # Load a preferred checkpoint
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
                    X, _ = train_dataset.get_batch(split)
                    with ctx:
                        loss = model(X, targets=None, return_type='loss', return_all_heads=False, loss_type='single_token')
                    losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
        
    history, token_history = [], []
    t0 = time.time()
    
    if args.init_from == 'resume':
        history = ckpt['history']
        tokens_seen = history[-1]['tokens_seen']
        
    # # Get the first batch
    X, y = train_dataset.get_batch('train')
    logging.info("Entering training loop...")
    # Training loop (train tokens / tokens_per_step = num of steps = 545002725 / 262144 ~ 2079)
    # Time per step ~ 20s, 20s * 2079 steps / 60^2 = 11.5 hours 
    while step < args.max_iters:
        # Get the current learning rate (either restored or new)
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Step {step}, Learning Rate: {current_lr}")

        # Save grad and activation history 
        if step % args.log_interval == 0:       
            all_hook_names = list(model.hook_dict.keys())
            cache = model.add_caching_hooks(names_filter=all_hook_names, incl_bwd=True) 
            grads_for_vis = [
                re.compile(r'^(hook_mtp_heads)\.(\d+)_grad$'),
                re.compile(r'^(blocks\.(\d+)\.hook_resid_post)_grad$'),
                re.compile(r'^(blocks\.(\d+)\.hook_resid_mid)_grad$')
            ]
        # Evaluate the loss on train/val sets and write checkpoints
        if step % args.eval_interval == 0:
            losses = estimate_loss()
            tokens_seen_per_eval_step = args.batch_size * args.dataset_config.block_size * args.eval_interval
            logging.info(f"Eval Step {step} across {args.eval_iters} batches ({tokens_seen_per_eval_step / 1e6:.1f}M Tokens): Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")
            logging.info(f"Generating text: \n{generate_text(model, args.device, ctx, max_length=124)}")
            if losses['val'] < best_val_loss or args.always_save_checkpoint:
                best_val_loss = losses['val']
                save_checkpoint(args.save_dir, model, optimizer, scheduler, step, best_val_loss, history, cache)
        
        # Do only an eval step
        if step == 0 and args.eval_only:
            break
        
        # Gradient accumulation loop 32 (batch_size) * (32) (gradient_accumulation_steps) = 1024 effective batch_size per optimizer step 
        # tokens per effective batch = 262,144
        logging.info(f"Entering gradient accumulation loop for micro-step {step}")
        for micro_step in range(args.gradient_accumulation_steps):
            head_losses = {}
    
            # Save token history:
            token_history.extend(X.flatten().tolist())

            if args.init_from == 'resume':
                total_tokens_seen = tokens_seen + len(token_history)
            
            else: 
                total_tokens_seen = len(token_history)
                    
            # 1) Forward pass through the shared trunk
            #    shape: (batch, seq_len, d_model)
            with ctx: 
                prompt_len = X.size(1) - args.n_future  # For example, if block_size is 512 and n_future is 4, prompt_len becomes 508.
                prompt = X[:, :prompt_len]              # The trunk sees only the prompt tokens.
                trunk_out = model.shared_trunk(prompt)

            # 2) "Chained" forward pass through each MTP head in order
            #    We store each head's output so we can do backward in reverse
            chained = [trunk_out]

            for i in range(args.n_future):
                # Head i sees the output of Head (i-1)
                # shape still: (batch, seq_len, d_model)
                with ctx: 
                    next_resid = model.hook_mtp_heads[i](model.mtp_heads[i](chained[-1]))
                    # Optionally do a LN
                    next_resid = model.ln_f(next_resid)
                    chained.append(next_resid)

            # 3) For each head i in reverse order, compute the cross-entropy
            #    and do partial backward. 
            #    Because each head is built on top of the previous, we must
            #    retain the graph until the last head has been backproped.
            total_loss = 0.0
            for i in reversed(range(args.n_future)):
                with ctx: 
                    # Unembed and compute logits
                    logits_i = model.hook_unembed(model.unembed(chained[i+1]))  # shape: (batch, seq_len, vocab_size)

                    # We want the token at position (prompt_len + i)
                    # i.e. the i-th future token in the sequence
                    pred_logits = logits_i[:, -1, :]  # (B, vocab_size)
                    target = y[:, prompt_len + i]                 # (B,)

                    # Calculate the loss for head i
                    loss_i = F.cross_entropy(pred_logits.reshape(-1, pred_logits.size(-1)), target.reshape(-1), ignore_index=model.tokenizer.pad_token_id)
                    
                    if micro_step % args.eval_interval == 0:
                        logging.info(f"Final Micro Step {micro_step + args.gradient_accumulation_steps}:  MTP Head {i} Loss: {loss_i.item():.2f}")
                
                loss_i_scaled = loss_i / args.gradient_accumulation_steps    
                
                head_losses[f"step_{step}_mtp_head_{i}_loss"] = loss_i_scaled.item()
                total_loss += loss_i.detach()

                # Partial backward:
                # We retain the graph if there are still earlier heads to process.
                scaler.scale(loss_i_scaled.backward(retain_graph=(i > 0)))

                del logits_i, pred_logits, loss_i, loss_i_scaled, target
                gc.collect()
                torch.cuda.empty_cache()

        del chained, trunk_out
        gc.collect()
        torch.cuda.empty_cache()
        # Get the next batch
        X, y = train_dataset.get_batch('train')  
        t1 = time.time()
        dt1 = t1 - t0
        logging.info(f"Step {step} | Total Loss: {total_loss.item():.2f} | Average Loss: {total_loss.item() / args.n_future:.2f} | Time: {dt1:.2f}s")
        # Log gradients before clipping
        if step % args.log_interval == 0:
            plot_grad_flow(cache, grads_for_vis, step)
            log_acts_and_grads(cache, step)
        
        # 4. Gradient clipping
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            norm = clip_grad_norm_(model.parameters(), args.grad_clip) # norm = accumulated gradient norm *before* clipping

        # 5. Optimizer step  
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
 
        # Log final loss
        average_loss_per_token = total_loss / model.n_future
        if step % args.log_interval == 0:
            # unscale loss for logging
            # loss_value = total_loss.item() * args.gradient_accumulation_steps / model.n_future
            logging.info(f"Step {step}: Loss {average_loss_per_token:.2f}, Norm: {norm:.4f}, Tokens Seen: {total_tokens_seen/ 1e6:.1f}M") 
        
        curr_history = {
                'step': step,
                'loss': average_loss_per_token.item(),
                'lr': current_lr,
                'head_losses': head_losses,
                'tokens_seen': len(token_history),
            }
        if step % args.eval_interval == 0:
            y_bigram = y[:, :1]  
            y_trigram = y[:, :2]  
            y_fourgram = y[:, :3]
                
            # Compute n-gram accuracies
            with torch.no_grad():
                logits = model.forward(X[:, :prompt_len])[:, -1, :, :]  # (batch, n_future, vocab_size)
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
    save_checkpoint(args.save_dir, model, optimizer, scheduler, step, best_val_loss, history, cache)
    logging.info(f"Training complete.") 