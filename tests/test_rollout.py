import torch
from twig.evaluation.rollout import autoregressive_rollout


class Repeat(torch.nn.Module):
    def forward(self, x): return x[:, -2:]


def test_rollout_truncates_to_requested_steps():
    result = autoregressive_rollout(Repeat(), torch.zeros(2, 4, 3, 1), steps=5)
    assert result.shape == (2, 5, 3, 1)
