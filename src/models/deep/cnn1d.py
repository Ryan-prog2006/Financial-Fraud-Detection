"""PyTorch 1D CNN model architecture for FinShield sequential fraud detection."""

import torch
import torch.nn as nn

class CNN1DFraudDetector(nn.Module):
    """1D CNN Classifier for sequential transaction fraud detection."""

    def __init__(self, input_size: int, dropout: float = 0.3):
        """Initializes neural network layers."""
        super().__init__()
        # Conv1d layers with padding
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input sequence tensor of shape (batch, seq_len, input_size).

        Returns:
            Logit predictions tensor of shape (batch, 1).
        """
        # Permute input from (batch, seq_len, input_size) to (batch, input_size, seq_len) for Conv1d
        x = x.permute(0, 2, 1)
        
        # Apply layers
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        
        x = self.pool(x)       # (batch, 128, 1)
        x = x.squeeze(-1)      # (batch, 128)
        
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)
