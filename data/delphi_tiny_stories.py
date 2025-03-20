# dataset: https://huggingface.co/datasets/delphi-suite/stories
import os
import pickle
from tqdm import tqdm
from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("delphi-suite/stories-tokenizer")
num_proc = 16

num_proc_load_dataset = num_proc

def tokenization(x):
    ids = tokenizer.encode(x['story'], add_special_tokens=True)
    ids.append(tokenizer.eos_token_id)
    out = {'ids': ids, 'len': len(ids)}
    return out

if __name__ == '__main__':
    # Directory setup
    save_dir = os.path.dirname(os.path.abspath(__file__).replace('data', 'data/delphi_stories'))
    os.makedirs(save_dir, exist_ok=True)

    dataset = load_dataset("delphi-suite/stories")
    print(f"Dataset columns: {dataset.column_names}")
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

    # tokenize the dataset
    tokenized = split_dataset.map(
        tokenization,
        remove_columns=['story'],
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

        # Load the tokenized data
        data = np.memmap(filename, dtype=np.uint16, mode='r')
        print(f"Length of dataset in characters: {len(data):,}")
        
        decoded_text = tokenizer.decode(data.tolist())

        text_tokens = decoded_text.split()
        # Compute n-gram statistics
        unigrams, bigrams, trigrams, fourgrams, bigrams_cond, trigrams_cond, fourgrams_cond = compute_ngrams(text_tokens)

        # Create mappings
        vocab_size = len(unigrams)
        stoi = {i: i for i in range(vocab_size)}
        itos = {i: i for i in range(vocab_size)}

        # Save metadata
        meta = {
            'vocab_size': vocab_size,
            'itos': itos,
            'stoi': stoi,
            'unigrams': unigrams,
            'bigrams': bigrams,
            'trigrams': trigrams,
            'fourgrams': fourgrams,
            'bigrams_cond': bigrams_cond,
            'trigrams_cond': trigrams_cond,
            'fourgrams_cond': fourgrams_cond
        }

        # Save metadata to a file
        meta_filename = os.path.join(data_dir, f"{split}_meta.pkl")
        with open(meta_filename, "wb") as f:
            pickle.dump(meta, f)

        print(f"Saved n-gram metadata to {meta_filename}")
