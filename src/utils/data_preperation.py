import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torch.nn.utils.rnn import pad_sequence

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
        game_seq = self.game[idx:idx + self.seq_len]
        value_seq = self.value[idx:idx + self.seq_len]
        policy_seq = self.policy[idx:idx + self.seq_len]

        # If we are near the end and sequence is too short, pad it
        if game_seq.shape[0] < self.seq_len:
            pad_size = self.seq_len - game_seq.shape[0]
            game_seq = torch.cat([game_seq, torch.zeros((pad_size, *game_seq.shape[1:]))])
            value_seq = torch.cat([value_seq, torch.zeros((pad_size,))])
            policy_seq = torch.cat([policy_seq, torch.zeros((pad_size, policy_seq.shape[-1]))])

        return game_seq, value_seq, policy_seq


def custom_collate(batch):
    """
    Custom collate function to pad sequences in a batch dynamically.
    """
    game_seq, value_seq, policy_seq = zip(*batch)  # Unpack batch tuples

    # Convert lists of tensors to a padded tensor with batch_first=True
    game_seq = pad_sequence(game_seq, batch_first=True, padding_value=0)
    value_seq = pad_sequence(value_seq, batch_first=True, padding_value=0)
    policy_seq = pad_sequence(policy_seq, batch_first=True, padding_value=0)

    return game_seq, value_seq, policy_seq

def chessDataLoader(game_file, value_file, policy_file, batch_size=64, num_workers=4):
    """
    Creates a DataLoader for chess tensor datasets with padding for variable batch sizes.
    """
    dataset = ChessTensorDataset(game_file, value_file, policy_file)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),  # Ensure sequential loading
        num_workers=num_workers,
        drop_last=False,  # Ensures all data is included
        collate_fn=custom_collate  # Handles padding dynamically
    )
