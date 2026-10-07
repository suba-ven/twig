from pathlib import Path
import numpy as np
import torch

from .rollout import autoregressive_rollout


def predict(model, loader, device="cpu"):
    model.to(device); outputs, targets = [], []
    with torch.inference_mode():
        for x, y, *_ in loader:
            outputs.append(model(x.to(device)).cpu().numpy()); targets.append(y.numpy())
    return np.concatenate(outputs), np.concatenate(targets)


def save_predictions(path, predictions, targets=None, **arrays):
    payload = {"predictions": predictions, **arrays}
    if targets is not None: payload["targets"] = targets
    Path(path).parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, **payload)
