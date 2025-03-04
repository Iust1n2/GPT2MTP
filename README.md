# GPT2MTP

To train the model run the following CLI command: 

```
python train.py \
    model_config.n_ctx=512 \
    n_future=4 \
    batch_size=32 \
    gradient_accumulation_steps=3 * 32 \
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
    data_dir="data/open_web_text_10k/" \
    save_dir="checkpoints/open_web_text_10k/" \
    always_save_checkpoint=True \
    eval_only=False \
    eval_interval=500 \
    eval_iters=200 \
    device="cuda" \
    dtype="bfloat16" \
    compile=False > log_train_owt_10k.txt
```

## PyTorch utilities for training

1. Mixed Precision Training: 

- https://pytorch.org/docs/stable/notes/amp_examples.html

2. Compile: 

- https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html

3. Profiling: 

- https://pytorch.org/docs/stable/torch.compiler_profiling_torch_compile.html
- https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html