import torch
import torch.nn as nn

class ChessEngine(nn.Module):
    def __init__(self, num_lstm_layers=2, hidden_dim=128, num_classes=4672):  # 4672 legal chess moves
        super(ChessEngine, self).__init__()

        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(19, 64, kernel_size=3, padding=1),  # Conv Layer 1
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Conv Layer 2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample
        )

        # LSTM for Sequential Processing
        self.lstm = nn.LSTM(input_size=128 * 4 * 4,  # Flattened CNN output
                            hidden_size=hidden_dim,
                            num_layers=num_lstm_layers,
                            batch_first=True)

        # Output layer (Policy Network)
        self.fc = nn.Linear(hidden_dim, num_classes)  # Predict move probabilities

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.shape  # (B, T, C, H, W)

        # CNN Processing: Flatten time dimension and extract features
        cnn_out = self.cnn(x.view(batch_size * seq_len, channels, height, width))  # (B*T, C', H', W')
        cnn_out = cnn_out.view(batch_size, seq_len, -1)  # Reshape for LSTM (B, T, Features)

        # LSTM Processing
        lstm_out, _ = self.lstm(cnn_out)  # (B, T, Hidden_dim)

        # Final prediction from last LSTM output
        output = self.fc(lstm_out[:, -1, :])  # Take last timestep

        return output
