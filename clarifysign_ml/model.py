import torch
import torch.nn as nn


class ISLBiLSTM(nn.Module):
    """2-layer BiLSTM over 48x225 sequences, mean+max temporal pooling, linear head."""

    def __init__(self, n_classes, in_dim=225, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sd", torch.ones(in_dim))
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(4 * hidden, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, n_classes))

    def forward(self, x):
        x = (x - self.mu) / self.sd
        h, _ = self.lstm(x)
        return self.head(torch.cat([h.mean(1), h.max(1).values], 1))
