"""PyTorch BiLSTM model architecture for FinShield sequential fraud detection."""

import torch
import torch.nn as nn

class BiLSTMFraudDetector(nn.Module):
    """Bidirectional LSTM Classifier for sequential transaction fraud detection."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
        """Initializes neural network layers."""
        super().__init__()
        # PyTorch bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input sequence tensor of shape (batch, seq_len, input_size).

        Returns:
            Logit predictions tensor of shape (batch, 1).
        """
        # lstm_out shape: (batch, seq_len, hidden_size * 2)
        lstm_out, _ = self.lstm(x)
        
        # Take the last time step output: (batch, hidden_size * 2)
        last_out = lstm_out[:, -1, :]
        
        # Apply Batch Normalization, Dropout, Dense, ReLU, and output Linear projection
        x = self.batch_norm(last_out)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)
