# dataset: https://huggingface.co/datasets/Skylion007/openwebtext
# script: https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py
import os
from tqdm import tqdm
import pickle
import numpy as np
from datasets import load_dataset
from collections import Counter
from itertools import chain
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 16

num_proc_load_dataset = num_proc

BATCH_SIZE = 10**6

def save_to_bin(file_path, data):
    """Save numpy array to binary file."""
    np.array(data, dtype=np.uint16).tofile(file_path)

def tokenization(x):
    ids = tokenizer.encode(x['text'], add_special_tokens=True)
    ids.append(tokenizer.eos_token_id)
    out = {'ids': ids, 'len': len(ids)}
    return out

def compute_ngrams(data):
    print("Computing unigram, bigram, trigram, and fourgram frequencies...")

    # Compute raw n-gram frequencies
    unigrams = dict(Counter(data))
    bigrams = dict(Counter(zip(data, data[1:])))
    trigrams = dict(Counter(zip(data, data[1:], data[2:])))
    fourgrams = dict(Counter(zip(data, data[1:], data[2:], data[3:])))

    # Compute conditional frequencies
    bigrams_cond = {}
    for (w1, w2), count in bigrams.items():
        bigrams_cond.setdefault(w1, {})[w2] = count

    trigrams_cond = {}
    for (w1, w2, w3), count in trigrams.items():
        trigrams_cond.setdefault((w1, w2), {})[w3] = count

    fourgrams_cond = {}
    for (w1, w2, w3, w4), count in fourgrams.items():
        fourgrams_cond.setdefault((w1, w2, w3), {})[w4] = count

    return unigrams, bigrams, trigrams, fourgrams, bigrams_cond, trigrams_cond, fourgrams_cond

# Incremental update function for n-grams
def update_ngram_counts(batch_data, unigrams, bigrams, trigrams, fourgrams):
    new_unigrams, new_bigrams, new_trigrams, new_fourgrams, _, _, _ = compute_ngrams(batch_data)
    unigrams.update(new_unigrams)
    bigrams.update(new_bigrams)
    trigrams.update(new_trigrams)
    fourgrams.update(new_fourgrams)

if __name__ == '__main__':
    # Directory setup
    save_dir = os.path.dirname(os.path.abspath(__file__).replace('data', 'data/open_web_text'))
    os.makedirs(save_dir, exist_ok=True)

    # Load a subset of OpenWebText
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)
    print(f"Dataset columns: {dataset.column_names}")
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val
    
    # tokenize the dataset
    tokenized = split_dataset.map(
        tokenization,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )
    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(save_dir, os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 256

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Ensure shard index is within range
            if batch_idx >= len(dset):  
                break
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        print(f"Finished writing {split}.bin with {arr_len} tokens")

    # Path to saved bin files
    data_dir = os.path.dirname(os.path.abspath(__file__))
    for split in ['train', 'val']:
        filename = os.path.join(data_dir, f"{split}.bin")
        print(f"Processing {filename}...")

        # Load the tokenized data using memmap
        data = np.memmap(filename, dtype=np.uint16, mode='r')
        print(f"Length of dataset in tokens: {len(data):,}")

        # Initialize n-gram counters
        unigrams = Counter()
        bigrams = Counter()
        trigrams = Counter()
        fourgrams = Counter()

        # Process in batches
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i + BATCH_SIZE]
            decoded_text = tokenizer.decode(batch.tolist())
            text_tokens = decoded_text.split()

            # Update n-gram counts incrementally
            update_ngram_counts(text_tokens, unigrams, bigrams, trigrams, fourgrams)
            print(f"Processed batch {i // BATCH_SIZE + 1} / {len(data) // BATCH_SIZE + 1}")

        # Compute conditional n-grams based on the full counts
        bigrams_cond = {}
        for (w1, w2), count in bigrams.items():
            bigrams_cond.setdefault(w1, {})[w2] = count

        trigrams_cond = {}
        for (w1, w2, w3), count in trigrams.items():
            trigrams_cond.setdefault((w1, w2), {})[w3] = count

        fourgrams_cond = {}
        for (w1, w2, w3, w4), count in fourgrams.items():
            fourgrams_cond.setdefault((w1, w2, w3), {})[w4] = count

        # Create mappings
        vocab_size = len(unigrams)
        stoi = {i: i for i in range(vocab_size)}
        itos = {i: i for i in range(vocab_size)}

        # Save metadata
        meta = {
            'vocab_size': vocab_size,
            'itos': itos,
            'stoi': stoi,
            'unigrams': dict(unigrams),
            'bigrams': dict(bigrams),
            'trigrams': dict(trigrams),
            'fourgrams': dict(fourgrams),
            'bigrams_cond': bigrams_cond,
            'trigrams_cond': trigrams_cond,
            'fourgrams_cond': fourgrams_cond
        }

        # Save metadata to a file
        meta_filename = os.path.join(data_dir, f"{split}_meta.pkl")
        with open(meta_filename, "wb") as f:
            pickle.dump(meta, f)

        print(f"Saved n-gram metadata to {meta_filename}")

# train.bin is ~17GB, val.bin ~8.5MB
# train has ~9B tokens (9,035,582,198)
# val has ~4M tokens (4,434,897)
# m = np.memmap('train.bin', dtype=np.uint16, mode='r')