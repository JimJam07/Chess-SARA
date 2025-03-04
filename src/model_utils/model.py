import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessEngine(nn.Module):
    def __init__(self, embed_dim=128, lstm_hidden_dim=256, num_lstm_layers=2):
        super(ChessEngine   , self).__init__()

        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=19, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Reduce to 128 feature maps
        )

        # LSTM for sequential learning
        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers, batch_first=True)

        # Value Head (Predicts position evaluation)
        self.value_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # Normalize to [-1,1]
        )

        # Policy Head (Predicts move embedding)
        self.policy_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, embed_dim),
            nn.ReLU()
        )

    def forward(self, board_tensors):
        """
        board_tensors: Shape (batch_size, seq_length, 19, 8, 8)
        """
        batch_size, seq_length, _, _, _ = board_tensors.shape

        # Reshape for CNN processing
        board_tensors = board_tensors.view(batch_size * seq_length, 19, 8, 8)

        # CNN Feature Extraction
        cnn_features = self.cnn(board_tensors)  # (batch * seq_length, 128, 1, 1)
        cnn_features = cnn_features.view(batch_size, seq_length, -1)  # Reshape: (batch, seq_length, 128)

        # LSTM Processing
        lstm_out, _ = self.lstm(cnn_features)  # Output shape: (batch, seq_length, lstm_hidden_dim)

        # Value & Policy Heads applied to **each timestep**
        value = self.value_head(lstm_out).squeeze(-1)  # Shape: (batch, seq_length)
        policy = self.policy_head(lstm_out)  # Shape: (batch, seq_length, 128)

        return value, policy  # Preserve seq_length dimension
