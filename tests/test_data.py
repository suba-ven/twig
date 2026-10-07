import numpy as np
from twig.data.common import SlidingWindowDataset, scenario_split


def test_splits_are_disjoint_and_complete():
    parts = scenario_split(20, seed=42)
    assert len(set(parts[0]) & set(parts[1])) == 0
    assert sorted(np.concatenate(parts).tolist()) == list(range(20))


def test_sliding_windows_have_expected_shapes():
    ds = SlidingWindowDataset(np.zeros((3, 12, 5)), history=4, horizon=3)
    x, y = ds[0]
    assert x.shape == (4, 5, 1) and y.shape == (3, 5, 1)
