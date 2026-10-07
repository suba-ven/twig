from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def scenario_split(n: int, fractions=(0.7, 0.15, 0.15), seed: int = 42):
    """Return deterministic, leakage-free train/validation/test scenario IDs."""
    if n < 3 or not np.isclose(sum(fractions), 1.0):
        raise ValueError("Need >=3 scenarios and fractions summing to one")
    ids = np.random.default_rng(seed).permutation(n)
    n_train = int(n * fractions[0])
    n_val = int(n * fractions[1])
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


class SlidingWindowDataset(Dataset):
    def __init__(self, trajectories, history: int, horizon: int):
        self.x = torch.as_tensor(trajectories, dtype=torch.float32)
        if self.x.ndim == 3:
            self.x = self.x.unsqueeze(-1)
        self.history, self.horizon = int(history), int(horizon)
        self.index = [(s, t) for s in range(len(self.x))
                      for t in range(self.x.shape[1] - history - horizon + 1)]

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        s, t = self.index[i]
        return self.x[s, t:t+self.history], self.x[s, t+self.history:t+self.history+self.horizon]
