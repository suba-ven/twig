import torch
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features * 2
        out_features = out_features or in_features
        self.value = nn.Linear(in_features, hidden_features)
        self.gate = nn.Linear(in_features, hidden_features)
        self.output = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.output(self.value(x) * torch.nn.functional.silu(self.gate(x)))
