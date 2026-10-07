import torch
from twig.models.twig import RNNDirect3D, TimeCausalEncoder


def test_rnn_shape():
    model = RNNDirect3D(history=4, forecast_horizon=3, channels=2, hidden_size=16, layers=1, dropout=0)
    assert model(torch.randn(2, 4, 7, 2)).shape == (2, 3, 7, 2)


def test_causal_encoder_shape():
    model = TimeCausalEncoder(history=7, channels=2, bands=3)
    assert model(torch.randn(2, 7, 5, 2)).shape == (2, 5, 10)
