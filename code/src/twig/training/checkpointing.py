from pathlib import Path
import torch


def save_checkpoint(path, model, optimizer=None, **metadata):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": model.state_dict(), **metadata}
    if optimizer is not None: payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path, model, map_location="cpu", strict=True):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload.get("model_state_dict", payload), strict=strict)
    return payload
