import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt

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

def save_checkpoint(save_dir, model, optimizer, scheduler, step, best_val_loss, history, cache):
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
        'cache': cache
        # 'config': model.config
    }, os.path.join(save_dir, f'checkpoint_step_{step}.pt'))

def plot_grad_flow(cache, regex_filters=None, step=None, save=True):
    """
    Plots the mean, max, and norm of the activations stored in cache.
    Creates two plots:
    1. Mean-gradient and max-gradient (linear scale).
    2. Norm-gradient (log scale).
    """
    ave_grads = []
    max_grads = []
    norm_grads = []
    layers = []
    
    # Loop over all cached activations.
    for n in cache:
        p = cache[n]
        # If regex_filters are provided, only keep activations matching one of them.
        if regex_filters is not None:
            matched = False
            label = n  # default label
            for pattern in regex_filters:
                m = pattern.match(n)
                if m:
                    matched = True
                    # If the pattern has groups (as in our MTP heads example),
                    # we can construct a nicer label.
                    # Example: "hook_mtp_heads.3_grad" becomes "hook_mtp_heads (head 3)"
                    if m.lastindex is not None and m.lastindex >= 2:
                        prefix = m.group(1)
                        index = m.group(2)
                        label = f"{prefix} (head {index})"
                    break
            if not matched:
                continue
        else:
            label = n
        
        layers.append(label)
        if p is not None:
            # Extract mean, max, and norm for the current activation
            p_np = p.float().detach().cpu().numpy()
            # p_np_norm = p.float().norm().detach().cpu().numpy()
            p_np_norm = torch.norm(p.float(), p=2).detach().cpu().numpy().item()
            ave_grads.append(p_np.mean())
            max_grads.append(p_np.max())
            norm_grads.append(p_np_norm)
        else:
            ave_grads.append(0)
            max_grads.append(0)
            norm_grads.append(0)
    
    # Plot 1: Mean and Max gradients (Linear Scale)
    plt.figure(figsize=(12, 6))
    x = np.arange(len(max_grads))
    # bar_width = 0.35  # Width of each bar

    # # Plot max and mean as separate sets of bars with an offset
    # plt.bar(x - bar_width/2, max_grads, width=bar_width, alpha=0.7, color="c", label="max-gradient")
    # plt.bar(x + bar_width/2, ave_grads, width=bar_width, alpha=0.7, color="b", label="mean-gradient")

    # plt.yscale('log')  # Use log scale for the y-axis
    # plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    # plt.xticks(x, layers, rotation="vertical", fontsize=8)
    # plt.xlim(left=0, right=len(ave_grads))
    # plt.ylim(bottom=0, top=max(max_grads) * 1.2 if max_grads else 0.02)
    # plt.xlabel("Layers")
    # plt.ylabel("Gradient value")
    # plt.title("Backprop Gradients (Max and Mean)")
    # plt.grid(True)
    # plt.legend()

    # save_dir1 = "grad_flow/stats"
    # if not os.path.exists(save_dir1):
    #     os.makedirs(save_dir1)

    # # Save the max/mean plot
    # if step is not None and save:
    #     plt.savefig(f"{save_dir1}/grad_flow_stats_step_{step}.png")
    # else:
    #     plt.show()
    # plt.close()

    # Plot 2: Norm gradients (Log Scale)
    plt.figure(figsize=(12, 6))
    plt.bar(x, norm_grads, alpha=0.7, color="g", label="norm-gradient")

    plt.yscale('log')  # Use log scale for the y-axis
    plt.xticks(x, layers, rotation="vertical", fontsize=8)
    plt.xlim(left=0, right=len(norm_grads))
    plt.ylim(bottom=max(min(norm_grads) * 0.1, 1e-10), top=max(norm_grads) * 10)
    plt.xlabel("Activations")
    plt.ylabel("L2 Norm")
    plt.title("Backprop Gradients Norm")
    plt.grid(True, which="both", linestyle="--", linewidth=0.7)
    # plt.legend()

    save_dir2 = "train/train_misc/grad_flow/norm"
    if not os.path.exists(save_dir2):
        os.makedirs(save_dir2)
    plt.tight_layout()
    # Save the norm plot
    if step is not None and save:
        plt.savefig(f"{save_dir2}/grad_flow_norm_{step}.png")
    else: 
        plt.show()
    plt.close()

def log_acts_and_grads(cache, step):
    save_dir = 'train/train_misc/grad_flow'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_dir_activations = os.path.join(save_dir, f"activations/log_step_{step}")
    log_dir_gradients = os.path.join(save_dir, f"gradients/log_step_{step}")
    os.makedirs(log_dir_activations, exist_ok=True)
    os.makedirs(log_dir_gradients, exist_ok=True)
    # Compile regex pattern for gradients only
    regex_pattern = r'.*_grad$'
    grad_pattern = re.compile(regex_pattern)

    activations_log_entries = []
    gradients_log_entries = []

    for i, n in enumerate(cache):
        p = cache[n]
        if p is not None:
            p_np = p.float().detach().cpu().numpy()
            p_np_norm = float(torch.norm(p.float(), p=2).detach().cpu())

            # Check if it's a gradient using regex
            if grad_pattern.match(n):
                # Format the gradient log entry with actual values
                grad_entry = f"{n}: mean={p_np.mean()}, max={p_np.max()} std_abs={p_np.std()}, norm={p_np_norm:.6f}"
                gradients_log_entries.append(grad_entry)
            else:
                # Format the activation log entry
                activation_entry = f"{n}: mean={p_np.mean()}, max={p_np.max()}, std_abs={p_np.std()} norm={p_np_norm:.6f}"
                activations_log_entries.append(activation_entry)
        else:
            # Explicitly log None values
            if grad_pattern.match(n):
                gradients_log_entries.append(f"{n}: None")
            else:
                activations_log_entries.append(f"{n}: None")

    # Write all activation entries to the log file
    activation_log_file = os.path.join(log_dir_activations, "activations_log.txt")
    with open(activation_log_file, "a") as f:
        f.write("\n".join(activations_log_entries) + "\n")

    # Write all gradient entries to the log file
    gradient_log_file = os.path.join(log_dir_gradients, "gradients_log.txt")
    with open(gradient_log_file, "a") as f:
        f.write("\n".join(gradients_log_entries) + "\n")

