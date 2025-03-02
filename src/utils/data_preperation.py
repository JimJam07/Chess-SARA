import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import numpy as np


class ChessTensorDataset(Dataset):
    """
    Custom PyTorch Dataset for loading chess tensor data efficiently.
    """
    def __init__(self, tensor_file):
        """
        :params: tensor_file (str): Path to the saved tensor dataset (.pt file).
        """
        # Load data as a memory-mapped tensor (efficient for large datasets)
        self.data = torch.load(tensor_file, map_location="cpu")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def chessDataLoader(tensor_file, batch_size=64, shuffle=True, num_workers=4):
    """
    Creates a DataLoader for chess tensor datasets.

    Args:
        tensor_file (str): Path to saved tensor dataset.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle dataset.
        num_workers (int): Number of worker threads for data loading.

    Returns:
        DataLoader: PyTorch DataLoader instance.
    """
    dataset = ChessTensorDataset(tensor_file)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),  # Ensure sequential loading
        num_workers=num_workers,
        drop_last=False  # Ensures all data is included
    )



