import torch


def rmse_loss(prediction, target, eps=0.0):
    return torch.sqrt(torch.mean((prediction - target) ** 2) + eps)
