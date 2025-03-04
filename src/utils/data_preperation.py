import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import numpy as np


class ChessTensorDataset(Dataset):
    """
    Custom PyTorch Dataset for loading chess tensor data efficiently.
    """
    def __init__(self, game_file, value_file, policy_file, seq_len=10):
        """
        :params: tensor_file (str): Path to the saved tensor dataset (.pt file).
        """
        # Load data as a memory-mapped tensor (efficient for large datasets)
        self.game = torch.load(game_file, map_location="cpu")
        self.value = torch.load(value_file, map_location="cpu")
        self.policy = torch.load(policy_file, map_location="cpu")
        self.seq_len = seq_len

    def __len__(self):
        return len(self.game) - self.seq_len

    def __getitem__(self, idx):
        return self.game[idx:idx + self.seq_len], self.value[idx:idx + self.seq_len], self.policy[idx:idx + self.seq_len]


def chessDataLoader(game_file, value_file, policy_file, batch_size=64, shuffle=True, num_workers=4):
    """
    Creates a DataLoader for chess tensor datasets.

    Args:
        game_file (str): Path to saved tensor dataset.
        value_file (str): Path to the saved score dataset (.pt file)
        policy_file (str): Path to the moves dataset (.pt file)
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle dataset.
        num_workers (int): Number of worker threads for data loading.

    Returns:
        DataLoader: PyTorch DataLoader instance.
    """
    dataset = ChessTensorDataset(game_file, value_file, policy_file)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),  # Ensure sequential loading
        num_workers=num_workers,
        drop_last=False  # Ensures all data is included
    )



