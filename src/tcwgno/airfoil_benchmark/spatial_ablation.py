"""Five parameter-matched TC-WGNO spatial allocation experiments, without GINE."""

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from . import models


VARIANTS = {
    "deep40": dict(width=124, depth=40, n_scales=12, branch="none"),
    "polynomial": dict(width=248, depth=10, n_scales=8, branch="polynomial"),
    "attention": dict(width=248, depth=10, n_scales=8, branch="attention"),
    "deep_scales": dict(width=132, depth=20, n_scales=24, branch="none"),
    "shallow_wide": dict(width=556, depth=4, n_scales=4, branch="none"),
}

# Fixed-backbone controls: do not widen after removing a component.
CONTROLS = {
    "no_attention": dict(width=248, depth=10, n_scales=8, branch="none"),
    "no_wavelet": dict(width=248, depth=10, n_scales=0, branch="attention_only"),
    "raw_history": dict(width=248, depth=10, n_scales=8, branch="attention"),
    "raw_history_tc": dict(width=248, depth=10, n_scales=8, branch="attention"),
    "tc_bands3": dict(width=248, depth=10, n_scales=8, branch="attention", bands=3),
    "tc_bands8": dict(width=248, depth=10, n_scales=8, branch="attention", bands=8),
}


class RawHistoryEncoder(nn.Module):
    """Flatten each node's observed history for the learned input projection."""

    def forward(self, context):
        b, h, n, c = context.shape
        return context.permute(0, 2, 1, 3).reshape(b, n, h * c)


class AttentionOnlyBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, 4, dropout=0.0, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.mixer = models.SwiGLUMixer(width, dropout)

    def forward(self, x, modes, evals):
        z = self.attention_norm(x)
        update, _ = self.attention(z, z, z, need_weights=False)
        x = x + update
        return x + self.mixer(self.norm2(x))


def rescaled_laplacian(graph):
    """L_tilde = L - I = -D^-1/2 A D^-1/2, using lambda_max bound 2.

    Symmetrization and inverse-distance weights match the saved basis builder.
    No dense adjacency or truncated eigenbasis is used.
    """
    indices = graph.edge_index.cpu()
    weights = graph.edge_weight.cpu().float()
    indices = torch.cat((indices, indices.flip(0)), dim=1)
    weights = torch.cat((weights, weights)) * 0.5
    adjacency = torch.sparse_coo_tensor(
        indices, weights, (graph.n_nodes, graph.n_nodes)
    ).coalesce()
    row, col = adjacency.indices()
    degree = torch.zeros(graph.n_nodes).index_add_(0, row, adjacency.values())
    inv = degree.clamp_min(1e-12).rsqrt()
    values = -adjacency.values() * inv[row] * inv[col]
    return torch.sparse_coo_tensor(adjacency.indices(), values, adjacency.shape).coalesce()


class PolynomialOperator(nn.Module):
    """Four full-channel Chebyshev terms T_0 through T_3 on the full graph."""

    def __init__(self, width, laplacian):
        super().__init__()
        self.register_buffer("laplacian", laplacian, persistent=False)
        self.transforms = nn.Parameter(torch.randn(4, width, width) / math.sqrt(width))

    def propagate(self, x):
        b, n, w = x.shape
        # Sparse CUDA matmul stays FP32 even when the surrounding model uses AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            flat = x.float().permute(1, 0, 2).reshape(n, b * w)
            result = torch.sparse.mm(self.laplacian, flat)
            return result.reshape(n, b, w).permute(1, 0, 2)

    def forward(self, x):
        previous = x.float()
        current = self.propagate(previous)
        out = previous @ self.transforms[0] + current @ self.transforms[1]
        for order in (2, 3):
            following = 2 * self.propagate(current) - previous
            out = out + following @ self.transforms[order]
            previous, current = current, following
        return out


class PolynomialBlock(models.PaperGraphWNOBlock3D):
    def __init__(self, width, n_scales, dropout, laplacian):
        super().__init__(width, n_scales, dropout)
        self.polynomial = PolynomialOperator(width, laplacian)

    def forward(self, x, modes, evals):
        z = self.norm1(x)
        x = x + self.wavelet(z, modes, evals) + self.polynomial(z)
        return x + self.mixer(self.norm2(x))


class AttentionBlock(models.PaperGraphWNOBlock3D):
    def __init__(self, width, n_scales, dropout):
        super().__init__(width, n_scales, dropout)
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, 4, dropout=0.0, batch_first=True)

    def forward(self, x, modes, evals):
        x = x + self.wavelet(self.norm1(x), modes, evals)
        z = self.attention_norm(x)
        update, _ = self.attention(z, z, z, need_weights=False)
        x = x + update
        return x + self.mixer(self.norm2(x))


class CheckpointBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x, modes, evals):
        if self.training and torch.is_grad_enabled():
            return checkpoint(self.block, x, modes, evals, use_reentrant=False)
        return self.block(x, modes, evals)


def build_model(variant, graph, modes, evals, cfg, checkpoint_blocks=True):
    settings = {**VARIANTS, **CONTROLS}[variant]
    width, depth, scales = (settings[k] for k in ("width", "depth", "n_scales"))
    model = models.PaperTCWGNO3D(
        graph, evals, modes, cfg.history, cfg.forecast_horizon, cfg.channels,
        width=width, bands=settings.get("bands", 5), depth=depth, n_scales=scales, n_modes=128,
        dropout=0.05, enhanced_input=False, raw_history=variant == "raw_history_tc",
    )
    if settings["branch"] == "polynomial":
        laplacian = rescaled_laplacian(graph)
        model.blocks = nn.ModuleList([
            PolynomialBlock(width, scales, 0.05, laplacian) for _ in range(depth)
        ])
    elif settings["branch"] == "attention":
        model.blocks = nn.ModuleList([
            AttentionBlock(width, scales, 0.05) for _ in range(depth)
        ])
    elif settings["branch"] == "attention_only":
        model.blocks = nn.ModuleList([
            AttentionOnlyBlock(width, 0.05) for _ in range(depth)
        ])
    if variant == "raw_history":
        model.tc = RawHistoryEncoder()
        model.input_proj = nn.Linear(
            cfg.history * cfg.channels + graph.node_static.shape[1], width
        )
    if checkpoint_blocks:
        model.blocks = nn.ModuleList([CheckpointBlock(block) for block in model.blocks])
    return model
