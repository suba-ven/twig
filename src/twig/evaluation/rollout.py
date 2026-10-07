import torch


@torch.inference_mode()
def autoregressive_rollout(model, context, steps, history=None):
    model.eval(); history = history or context.shape[1]; blocks = []
    while sum(x.shape[1] for x in blocks) < steps:
        prediction = model(context)
        blocks.append(prediction)
        context = torch.cat((context, prediction), dim=1)[:, -history:]
    return torch.cat(blocks, dim=1)[:, :steps]
