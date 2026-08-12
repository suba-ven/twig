"""Operator baselines for the direct 14 -> 14 benchmark.

This module adds two spatial operator baselines that use exactly the same
history, static-node features, graph Laplacian modes, and direct 14-frame
prediction target as the SA TC-WGNO benchmark:

1. PlainGraphWNOBaseline
   A spatial Graph WNO stack with NO temporal-causal wavelet encoder.  The
   14 historical values are concatenated as ordinary node features.

2. GraphFNOBaseline
   An FNO-style graph spectral operator.  It performs mode-dependent feature
   mixing in the truncated graph-Laplacian eigenbasis, followed by pointwise
   mixing and a residual FFN.

Both models are capacity-matched to a supplied SA TC-WGNO reference parameter
count by choosing their width from a candidate range.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
import json
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from .direct14_config import Direct14Config
from .direct14_models import MLP, StaticGraphModule
from .direct14_registry import DirectModelSpec, count_parameters
from .graph import GraphStatic


def direct_plain_gwno_name() -> str:
    return "GWNO-D14"


def direct_graph_fno_name() -> str:
    return "GraphFNO-D14"


def _number_of_modes(
    modes: torch.Tensor,
    eigenvalues: torch.Tensor,
) -> int:
    """Infer the retained spectral-mode count from the eigenvalue vector."""
    if eigenvalues.ndim != 1:
        raise ValueError(
            "Expected eigenvalues with shape [K], got "
            f"{tuple(eigenvalues.shape)}."
        )

    n_modes = int(eigenvalues.numel())

    if modes.ndim != 2:
        raise ValueError(
            "Expected graph modes with shape [N,K] or [K,N], got "
            f"{tuple(modes.shape)}."
        )

    if n_modes not in modes.shape:
        raise ValueError(
            "The eigenvalue count does not match either dimension of modes: "
            f"modes={tuple(modes.shape)}, eigenvalues={tuple(eigenvalues.shape)}."
        )

    return n_modes


def _canonical_eigenvectors(
    modes: torch.Tensor,
    eigenvalues: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    """Return graph eigenvectors with canonical layout [N, K]."""
    n_modes = _number_of_modes(modes, eigenvalues)

    if tuple(modes.shape) == (n_nodes, n_modes):
        return modes

    if tuple(modes.shape) == (n_modes, n_nodes):
        return modes.transpose(0, 1).contiguous()

    raise ValueError(
        "Graph mode shape is inconsistent with the static graph: "
        f"expected [{n_nodes},{n_modes}] or [{n_modes},{n_nodes}], got "
        f"{tuple(modes.shape)}."
    )


class PlainGraphWNOBaseline(StaticGraphModule):
    """Spatial Graph WNO baseline without a temporal-causal encoder.

    The model consumes the full direct-14 context as per-node feature channels:
        [B, H, N, 1] -> [B, N, H + C_static] -> Graph WNO stack.

    There is no temporal decomposition, no causal temporal convolution, and no
    temporal wavelet band construction in this baseline.
    """

    def __init__(
        self,
        graph: GraphStatic,
        history: int,
        forecast_horizon: int,
        width: int,
        modes: torch.Tensor,
        eigenvalues: torch.Tensor,
        graph_wno_block: type[nn.Module],
        *,
        layers: int = 10,
        n_scales: int = 4,
        dropout: float = 0.05,
    ):
        super().__init__(graph)

        self.history = int(history)
        self.forecast_horizon = int(forecast_horizon)
        self.width = int(width)
        self.n_modes = _number_of_modes(modes, eigenvalues)

        # Preserve the original mode orientation because this is the exact
        # representation expected by the existing GraphWNOBlock3D code.
        self.register_buffer("U_modes", modes.detach().clone())
        self.register_buffer("evals_modes", eigenvalues.detach().clone())

        self.encoder = MLP(
            [
                self.history + self.node_static.shape[1],
                self.width,
                self.width,
            ],
            layer_norm=True,
        )

        self.blocks = nn.ModuleList(
            [
                graph_wno_block(
                    width=self.width,
                    n_modes=self.n_modes,
                    n_scales=int(n_scales),
                    dropout=float(dropout),
                )
                for _ in range(int(layers))
            ]
        )

        self.decoder = MLP(
            [self.width, self.width, self.forecast_horizon]
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        self.validate_context(context, self.history)

        batch = context.shape[0]
        history = context.squeeze(-1).permute(0, 2, 1)

        static = self.node_static.unsqueeze(0).expand(batch, -1, -1)
        x = self.encoder(torch.cat([history, static], dim=-1))

        for block in self.blocks:
            x = block(x, self.U_modes, self.evals_modes)

        delta = self.decoder(
            x.reshape(batch * self.n_nodes, self.width)
        ).reshape(batch, self.n_nodes, self.forecast_horizon)

        delta = delta.permute(0, 2, 1).contiguous().unsqueeze(-1)
        return context[:, -1:].expand_as(delta) + delta


class GraphSpectralConv(nn.Module):
    """FNO-like spectral mixing over a truncated graph-Laplacian basis.

    Given U_K in R^{N x K}, this layer computes

        X_hat = U_K^T X,
        Y_hat_k = X_hat_k W_k,
        Y = U_K Y_hat,

    where each retained graph frequency k has its own learnable feature-mixing
    matrix W_k.  This is the graph analogue of retaining Fourier modes and
    applying mode-dependent FNO weights.
    """

    def __init__(self, width: int, n_modes: int):
        super().__init__()

        self.width = int(width)
        self.n_modes = int(n_modes)

        scale = 1.0 / math.sqrt(float(self.width))
        self.weight = nn.Parameter(
            scale * torch.randn(self.n_modes, self.width, self.width)
        )
        self.bias = nn.Parameter(torch.zeros(self.width))

    def forward(
        self,
        x: torch.Tensor,
        eigenvectors: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected x with shape [B,N,C], got {tuple(x.shape)}."
            )

        batch, nodes, width = x.shape

        if width != self.width:
            raise ValueError(
                f"Expected feature width {self.width}, got {width}."
            )

        if tuple(eigenvectors.shape) != (nodes, self.n_modes):
            raise ValueError(
                "Expected eigenvectors with shape "
                f"[{nodes},{self.n_modes}], got {tuple(eigenvectors.shape)}."
            )

        U = eigenvectors.to(dtype=x.dtype)

        # [B, N, C] -> [B, K, C]
        spectral_coefficients = torch.einsum("nk,bnc->bkc", U, x)

        # Independent learned feature transformation for every graph mode.
        mixed_coefficients = torch.einsum(
            "bki,kio->bko",
            spectral_coefficients,
            self.weight.to(dtype=x.dtype),
        )

        # [B, K, C] -> [B, N, C]
        return torch.einsum("nk,bkc->bnc", U, mixed_coefficients) + self.bias


class GraphFNOBlock(nn.Module):
    """Residual graph-FNO block: graph spectral operator + local FFN."""

    def __init__(self, width: int, n_modes: int):
        super().__init__()

        self.operator_norm = nn.LayerNorm(width)
        self.spectral = GraphSpectralConv(width=width, n_modes=n_modes)
        self.pointwise = nn.Linear(width, width)

        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = MLP([width, 2 * width, width])

    def forward(
        self,
        x: torch.Tensor,
        eigenvectors: torch.Tensor,
    ) -> torch.Tensor:
        h = self.operator_norm(x)
        x = x + F.gelu(self.spectral(h, eigenvectors) + self.pointwise(h))
        return x + self.ffn(self.ffn_norm(x))


class GraphFNOBaseline(StaticGraphModule):
    """Direct 14-frame graph spectral FNO baseline.

    This is deliberately spatial only.  The history enters as ordinary node
    channels; no temporal-causal encoding or temporal wavelet features are
    included.
    """

    def __init__(
        self,
        graph: GraphStatic,
        history: int,
        forecast_horizon: int,
        width: int,
        modes: torch.Tensor,
        eigenvalues: torch.Tensor,
        *,
        layers: int = 10,
    ):
        super().__init__(graph)

        self.history = int(history)
        self.forecast_horizon = int(forecast_horizon)
        self.width = int(width)
        self.n_modes = _number_of_modes(modes, eigenvalues)

        self.register_buffer(
            "eigenvectors",
            _canonical_eigenvectors(
                modes=modes,
                eigenvalues=eigenvalues,
                n_nodes=self.n_nodes,
            ).detach().clone(),
        )

        self.encoder = MLP(
            [
                self.history + self.node_static.shape[1],
                self.width,
                self.width,
            ],
            layer_norm=True,
        )

        self.blocks = nn.ModuleList(
            [
                GraphFNOBlock(
                    width=self.width,
                    n_modes=self.n_modes,
                )
                for _ in range(int(layers))
            ]
        )

        self.decoder = MLP(
            [self.width, self.width, self.forecast_horizon]
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        self.validate_context(context, self.history)

        batch = context.shape[0]
        history = context.squeeze(-1).permute(0, 2, 1)

        static = self.node_static.unsqueeze(0).expand(batch, -1, -1)
        x = self.encoder(torch.cat([history, static], dim=-1))

        for block in self.blocks:
            x = block(x, self.eigenvectors)

        delta = self.decoder(
            x.reshape(batch * self.n_nodes, self.width)
        ).reshape(batch, self.n_nodes, self.forecast_horizon)

        delta = delta.permute(0, 2, 1).contiguous().unsqueeze(-1)
        return context[:, -1:].expand_as(delta) + delta


def _nearest_width(
    factory: Callable[[int], nn.Module],
    target_parameters: int,
    candidates: range,
) -> tuple[int, int]:
    """Choose the candidate width closest to the reference parameter count."""
    choices: list[tuple[int, int, int]] = []

    for width in candidates:
        model = factory(int(width))
        parameters = count_parameters(model)
        choices.append(
            (
                abs(int(parameters) - int(target_parameters)),
                int(width),
                int(parameters),
            )
        )
        del model

    if not choices:
        raise ValueError("At least one width candidate is required.")

    _, width, parameters = min(choices)
    return width, parameters


def build_direct14_operator_baseline_specs(
    cfg: Direct14Config,
    graph: GraphStatic,
    modes: torch.Tensor,
    eigenvalues: torch.Tensor,
    graph_wno_block: type[nn.Module],
    device: torch.device,
    *,
    target_parameters: int,
    layers: int = 10,
    gwno_scales: int = 4,
    gwno_dropout: float = 0.05,
    gwno_width_candidates: range = range(16, 129, 4),
    graph_fno_width_candidates: range = range(4, 33),
) -> tuple[dict[str, DirectModelSpec], dict[str, dict[str, int]]]:
    """Build capacity-matched plain GWNO and graph-FNO D14 specifications."""

    if cfg.history != 14 or cfg.forecast_horizon != 14:
        raise ValueError(
            "These operator baselines require "
            "history=forecast_horizon=14."
        )

    factories: dict[str, Callable[[int], nn.Module]] = {
        direct_plain_gwno_name(): lambda width: PlainGraphWNOBaseline(
            graph=graph,
            history=cfg.history,
            forecast_horizon=cfg.forecast_horizon,
            width=width,
            modes=modes,
            eigenvalues=eigenvalues,
            graph_wno_block=graph_wno_block,
            layers=layers,
            n_scales=gwno_scales,
            dropout=gwno_dropout,
        ),
        direct_graph_fno_name(): lambda width: GraphFNOBaseline(
            graph=graph,
            history=cfg.history,
            forecast_horizon=cfg.forecast_horizon,
            width=width,
            modes=modes,
            eigenvalues=eigenvalues,
            layers=layers,
        ),
    }

    candidate_ranges = {
        direct_plain_gwno_name(): gwno_width_candidates,
        direct_graph_fno_name(): graph_fno_width_candidates,
    }

    specs: dict[str, DirectModelSpec] = {}
    matched: dict[str, dict[str, int]] = {}

    for name, factory in factories.items():
        width, parameters = _nearest_width(
            factory=factory,
            target_parameters=target_parameters,
            candidates=candidate_ranges[name],
        )

        matched[name] = {
            "width": int(width),
            "parameters": int(parameters),
            "target_parameters": int(target_parameters),
        }

        specs[name] = DirectModelSpec(
            name=name,
            factory=lambda factory=factory, width=width: (
                factory(width).to(device)
            ),
            metadata={
                "family": name.replace("-D14", ""),
                "history": int(cfg.history),
                "forecast_horizon": int(cfg.forecast_horizon),
                "width": int(width),
                "layers": int(layers),
                "n_modes": int(_number_of_modes(modes, eigenvalues)),
                "training_mode": "direct_14_to_14",
                "temporal_encoding": "raw_history_only",
                "matched_target_parameters": int(target_parameters),
            },
        )

    return specs, matched


@dataclass(frozen=True)
class Direct14TrainConfig:
    """Training hyperparameters for the new operator baselines."""

    epochs: int = 50
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    patience: int = 12
    grad_clip_norm: float | None = 1.0


def _unpack_direct14_batch(
    batch: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract (context, target) from tuple/list or mapping DataLoader batches."""
    if isinstance(batch, Mapping):
        context = next(
            (
                batch[key]
                for key in ("context", "x", "inputs", "history")
                if key in batch
            ),
            None,
        )
        target = next(
            (
                batch[key]
                for key in ("target", "y", "targets", "future")
                if key in batch
            ),
            None,
        )
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        context, target = batch[0], batch[1]
    else:
        raise TypeError(
            "Each batch must be either (context, target) or a mapping containing "
            "context/x/inputs/history and target/y/targets/future."
        )

    if not isinstance(context, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("Context and target must both be torch.Tensor objects.")

    return context, target


def _to_device(
    batch: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    context, target = _unpack_direct14_batch(batch)
    return (
        context.to(device, non_blocking=True),
        target.to(device, non_blocking=True),
    )


def _validate_prediction_shape(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if tuple(prediction.shape) != tuple(target.shape):
        raise ValueError(
            "Model prediction and target shapes must match for direct 14->14 "
            f"training, got prediction={tuple(prediction.shape)} and "
            f"target={tuple(target.shape)}."
        )


@torch.no_grad()
def evaluate_direct14_model(
    model: nn.Module,
    loader: Any,
    device: torch.device,
) -> dict[str, float | list[float]]:
    """Compute aggregate, final-frame, and per-horizon MSE on a loader."""
    model.eval()

    total_sse = 0.0
    total_elements = 0
    final_sse = 0.0
    final_elements = 0
    horizon_sse: torch.Tensor | None = None
    horizon_elements = 0

    for batch in loader:
        context, target = _to_device(batch, device)
        prediction = model(context)
        _validate_prediction_shape(prediction, target)

        squared_error = (prediction - target).square()
        total_sse += float(squared_error.sum().item())
        total_elements += int(squared_error.numel())

        final_error = squared_error[:, -1]
        final_sse += float(final_error.sum().item())
        final_elements += int(final_error.numel())

        this_horizon_sse = squared_error.sum(dim=(0, 2, 3)).detach().cpu()
        horizon_sse = (
            this_horizon_sse
            if horizon_sse is None
            else horizon_sse + this_horizon_sse
        )
        horizon_elements += int(
            squared_error.shape[0]
            * squared_error.shape[2]
            * squared_error.shape[3]
        )

    if total_elements == 0:
        raise ValueError("Cannot evaluate an empty data loader.")

    assert horizon_sse is not None
    return {
        "mse": total_sse / total_elements,
        "final_mse": final_sse / final_elements,
        "per_horizon_mse": (
            horizon_sse / float(horizon_elements)
        ).tolist(),
    }


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def fit_direct14_model(
    model: nn.Module,
    train_loader: Any,
    validation_loader: Any,
    device: torch.device,
    *,
    config: Direct14TrainConfig = Direct14TrainConfig(),
) -> tuple[nn.Module, list[dict[str, float | int]]]:
    """Train a direct 14->14 model and restore the best validation checkpoint."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )

    best_validation_mse = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        training_sse = 0.0
        training_elements = 0

        for batch in train_loader:
            context, target = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            prediction = model(context)
            _validate_prediction_shape(prediction, target)

            loss = F.mse_loss(prediction, target)
            loss.backward()

            if config.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(config.grad_clip_norm),
                )

            optimizer.step()

            squared_error = (prediction.detach() - target).square()
            training_sse += float(squared_error.sum().item())
            training_elements += int(squared_error.numel())

        if training_elements == 0:
            raise ValueError("Cannot train on an empty data loader.")

        validation_metrics = evaluate_direct14_model(
            model=model,
            loader=validation_loader,
            device=device,
        )

        train_mse = training_sse / training_elements
        validation_mse = float(validation_metrics["mse"])
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "validation_mse": validation_mse,
                "validation_final_mse": float(validation_metrics["final_mse"]),
            }
        )

        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_state = _cpu_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(config.patience):
            break

    if best_state is None:
        raise RuntimeError("No valid validation checkpoint was produced.")

    model.load_state_dict(best_state)
    return model, history


def _torch_load(path: str | Path, device: torch.device) -> Any:
    """Use weights_only=False when supported, while retaining old PyTorch compatibility."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _extract_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                return candidate

        if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
            return payload

    raise ValueError(
        "Checkpoint must be a state dict or contain one of: "
        "model_state_dict, state_dict, model."
    )


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    strict: bool = True,
) -> Any:
    """Load either a raw state dict or a common wrapped checkpoint format."""
    payload = _torch_load(checkpoint_path, device)
    state_dict = _extract_state_dict(payload)

    # Supports checkpoints saved through DistributedDataParallel.
    cleaned_state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(cleaned_state_dict, strict=strict)
    return payload


def fit_or_load_direct14_model(
    model: nn.Module,
    train_loader: Any,
    validation_loader: Any,
    device: torch.device,
    checkpoint_path: str | Path,
    *,
    config: Direct14TrainConfig = Direct14TrainConfig(),
) -> tuple[nn.Module, list[dict[str, float | int]]]:
    """Load a cached baseline checkpoint, or train and persist it once."""
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.exists():
        payload = load_model_checkpoint(model, checkpoint_path, device)
        history = (
            payload.get("history", [])
            if isinstance(payload, Mapping)
            else []
        )
        return model, list(history)

    model, history = fit_direct14_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
        config=config,
    )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": _cpu_state_dict(model),
            "history": history,
            "train_config": asdict(config),
        },
        checkpoint_path,
    )

    return model, history


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def run_operator_benchmark(
    *,
    pretrained_sa_name: str,
    pretrained_sa_model: nn.Module,
    baseline_specs: Mapping[str, DirectModelSpec],
    train_loader: Any,
    validation_loader: Any,
    test_loader: Any,
    device: torch.device,
    output_dir: str | Path,
    train_config: Direct14TrainConfig = Direct14TrainConfig(),
) -> tuple[
    dict[str, dict[str, float | list[float]]],
    dict[str, list[dict[str, float | int]]],
]:
    """Evaluate a pretrained SA TC-WGNO and train/evaluate operator baselines.

    The SA TC-WGNO model is never altered.  Each new baseline is trained from
    scratch once, then cached in ``output_dir/checkpoints`` for later runs.
    """
    output_dir = Path(output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    pretrained_sa_model = pretrained_sa_model.to(device)

    results: dict[str, dict[str, float | list[float]]] = {
        pretrained_sa_name: evaluate_direct14_model(
            model=pretrained_sa_model,
            loader=test_loader,
            device=device,
        )
    }
    histories: dict[str, list[dict[str, float | int]]] = {}

    for name, spec in baseline_specs.items():
        model = spec.factory()
        model, history = fit_or_load_direct14_model(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            device=device,
            checkpoint_path=(checkpoints_dir / f"{_safe_filename(name)}.pt"),
            config=train_config,
        )

        histories[name] = history
        results[name] = evaluate_direct14_model(
            model=model,
            loader=test_loader,
            device=device,
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (output_dir / "operator_benchmark_results.json").open("w") as handle:
        json.dump(results, handle, indent=2)

    with (output_dir / "operator_benchmark_histories.json").open("w") as handle:
        json.dump(histories, handle, indent=2)

    return results, histories
