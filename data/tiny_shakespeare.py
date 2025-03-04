# taken from https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare_char/prepare.py
import os
import pickle
import requests
import numpy as np
from collections import Counter, defaultdict
from itertools import chain
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def save_to_bin(file_path, data):
    """Save numpy array to binary file."""
    np.array(data, dtype=np.uint16).tofile(file_path)

def tokenization(x):
    ids = tokenizer.encode(x['text'], add_special_tokens=True)
    ids.append(tokenizer.eos_token_id)
    out = {'ids': ids, 'len': len(ids)}
    return out

def compute_ngrams(data):
    print("Computing unigram, bigram, trigram and fourgram frequencies...")

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

if __name__ == '__main__':
    save_dir = os.path.dirname(os.path.abspath(__file__).replace('data', 'data/tiny_shakespeare'))
    os.makedirs(save_dir, exist_ok=True)
    # download the tiny shakespeare dataset
    input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
    if not os.path.exists(input_file_path):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w') as f:
            f.write(requests.get(data_url).text)

    with open(input_file_path, 'r') as f:
        data = f.read()
    print(f"length of dataset in characters: {len(data):,}")

    # get all the unique characters that occur in this text
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("all the unique characters:", ''.join(chars))
    print(f"vocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    def encode(s):
        return [stoi[c] for c in s] # encoder: take a string, output a list of integers
    def decode(l):
        ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # create the train and test splits
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    # encode both to integers
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

    # TODO adapt to owt dataset meta data
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