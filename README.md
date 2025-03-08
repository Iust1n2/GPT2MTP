# GPT2MTP

To train the model run the following CLI command: 

```
python train.py \
    n_future=4 \
    batch_size=32 \
    gradient_accumulation_steps=$((2 * 16)) \
    max_iters=10850 \
    lr_decay_iters=10850 \
    max_lr=3e-4 \
    decay_lr=True \
    warmup_steps=1000 \
    min_lr=1e-5 \
    weight_decay=0.1 \
    betas="[0.9, 0.95]" \
    grad_clip=1.0 \
    log_interval=1 \
    init_from="scratch" \
    data_dir="data/open_web_text/" \
    save_dir="checkpoints/open_web_text/" \
    always_save_checkpoint=False \
    eval_only=False \
    eval_interval=100 \
    eval_ngrams=False \
    eval_iters=200 \
    device="cuda" \
    dtype="bfloat16" \
    compile=True > log_train_owt.txt
```

## PyTorch utilities for training

1. Mixed Precision Training: 

- [Main Docs](https://pytorch.org/docs/stable/notes/amp_examples.html)
- [Gradient Scaling](https://pytorch.org/docs/stable/amp.html#gradient-scaling)

2. Compile: 

- https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html

3. Profiling: 

- https://pytorch.org/docs/stable/torch.compiler_profiling_torch_compile.html
- https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

The following save a trace of the forward pass through the mtp heads in a json that can be loaded at `chrome://tracing`: 
```
with profile(activities=activities, profile_memory=True, record_shapes=True) as prof2:
    with record_function("fwd_mtp_head"):
        mtp_resid_post = model.hook_mtp_heads[i](model.mtp_heads[i](d))    # (batch, seq_length, d_model)
# print(prof2.key_averages().table(sort_by=sort_by_keyword, row_limit=10))
prof2.export_chrome_trace(f"{profile_dir}/trace_fwd_mtp.json")
```

or just inline print table with: 

`print(prof2.key_averages().table(sort_by=sort_by_keyword, row_limit=10))`