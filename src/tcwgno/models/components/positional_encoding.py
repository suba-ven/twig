import math
import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, width, max_length=512):
        super().__init__()
        pos = torch.arange(max_length).float().unsqueeze(1)
        rate = torch.exp(torch.arange(0, width, 2).float() * (-math.log(10_000) / width))
        table = torch.zeros(max_length, width)
        table[:, 0::2], table[:, 1::2] = torch.sin(pos * rate), torch.cos(pos * rate[:width // 2])
        self.register_buffer("table", table, persistent=False)

    def forward(self, x): return x + self.table[:x.shape[-2]]
