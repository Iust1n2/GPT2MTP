from dataclasses import dataclass
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

@dataclass
class DatasetConfig():
    """Configuration for the dataset."""
    data_dir: str = None
    split: str = None
    batch_size: int = 8
    block_size: int = 512  # block_size + n_future ≤ max_context_length (1024 for all GPT-2's // 8)
    n_future: int = 4
    device : str = 'cuda' if torch.cuda.is_available() else 'cpu'

class OpenWebTextDataset(Dataset):
    def __init__(self, args: DatasetConfig, split: str):
        """
        PyTorch Dataset to load encoded OpenWebText data with support for predicting future tokens.

        Args:
            data_dir (str): Directory containing `train.bin`, `val.bin`, and `meta.pkl`.
            split (str): Either 'train' or 'val'.
            batch_size (int): Length of the input token sequence.
            n_future (int): Number of future tokens to predict (target is shifted by this amount).
        """
        self.args = args
        self.batch_size = args.batch_size
        self.block_size = args.block_size
        assert split in ['train', 'val'], "split must be 'train' or 'val'"
        assert self.args.n_future >= 1, "n_future must be at least 1"

        # Load binary token data with memory mapping
        data_path = os.path.join(self.args.data_dir, f'{split}.bin')
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')

    def __len__(self):
        # Ensure there are enough tokens for input + n_future shift
        return len(self.data) - self.args.batch_size - self.args.n_future + 1
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx:idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1:idx + 1 + self.block_size].astype(np.int64))
        return x, y

    def get_batch(self, split):
        """
        Returns:
            x (torch.LongTensor): Input tokens of shape (batch_size,)
            y (torch.LongTensor): Future target tokens shifted by n_future of shape (batch_size,)
        We recreate np.memmap every batch to avoid a memory leak, as per
        https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        """
        if split == 'train':
            data = np.memmap(os.path.join(self.args.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(self.args.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - self.args.block_size, (self.args.batch_size,))
        x = torch.stack([torch.from_numpy((self.data[i:i + self.args.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i + 1:i + 1 + self.args.block_size]).astype(np.int64)) for i in ix])
        if self.args.device == 'cuda':
            device = self.args.device
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
    
    def get_dataloader(self, split):
        dataset = OpenWebTextDataset(self, split)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

if __name__ == '__main__':
    args = DatasetConfig(data_dir='data/open_web_text_10k', split='train')
    dataset = OpenWebTextDataset(args, 'train')
    effective_bsz = 32 * 96 * 124 # batch size * gradient_accumulation_steps * block_size
    print(f"Tokens in dataset: {len(dataset) / 1e6:.2f}M")
    print(f"Tokens seen per effective optimization step: {effective_bsz}")
    print(f"Effective steps to see all tokens: {len(dataset) / effective_bsz:.2f}")
    