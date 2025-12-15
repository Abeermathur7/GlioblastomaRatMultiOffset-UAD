
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch import Tensor

import torch.nn.functional as F

class ConvolutionalAE_upd(nn.Module):
    def __init__(self, seq_len=26):
        super().__init__()

        # after MaxPool1d(kernel_size=2), length → seq_len // 2
        self.seq_len = seq_len
        self.half_len = seq_len // 2   # 26 → 13

        # encoder
        self.conv_down1 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, stride=1, padding='same'),
            nn.GELU(),
            nn.Conv1d(8, 16, kernel_size=3, stride=1, padding='same'),
            nn.GELU(),
        )

        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)

        self.conv_down2 = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=3, stride=1, padding='same'),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding='same'),
            nn.GELU(),
            nn.Flatten(),      # → (batch, 32 * half_len)
        )

        # bottleneck
        self.fnn_down = nn.Sequential(
            nn.Linear(32 * self.half_len, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
        )
        self.fnn_up = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 32 * self.half_len),
            nn.ReLU(),
            nn.Unflatten(1, (32, self.half_len)),  # → (batch, 32, half_len)
        )

        # decoder
        self.conv_up1 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.max_unpool = nn.MaxUnpool1d(kernel_size=2, stride=2)
        self.conv_up2 = nn.Sequential(
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv1d(8, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        # x: (batch, 1, seq_len)
        x = self.conv_down1(x)            # → (batch, 16, seq_len)
        x, inds = self.max_pool(x)        # → (batch, 16, half_len)
        x = self.conv_down2(x)            # → (batch, 32*half_len)
        x = self.fnn_down(x)              # → (batch, 16)

        x = self.fnn_up(x)                # → (batch, 32, half_len)
        x = self.conv_up1(x)              # → (batch, 16, half_len)
        x = self.max_unpool(x, inds)      # → (batch, 16, seq_len)
        x = self.conv_up2(x)              # → (batch, 1, seq_len)
        return x

