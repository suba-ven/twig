from torch.optim.lr_scheduler import CosineAnnealingLR


def cosine_scheduler(optimizer, epochs, min_lr=1e-5):
    return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
