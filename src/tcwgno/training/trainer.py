from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from tcwgno.data.pflotran import (
    load_normalized_trajectory,
    stats_from_jsonable,
)
from tcwgno.models.tc_wgno import ModelSpec, count_parameters


@dataclass
class TrainConfig:
    history: int = 10
    forecast_horizon: int = 10
    rollout_steps: int = 65
    epochs: int = 30
    batch_size: int = 8
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    early_stopping_patience: int = 10
    grad_clip_norm: float = 1.0
    scheduler: str = "cosine"  # cosine or none
    amp: bool = True


@dataclass
class RunRecord:
    model: str
    label: str
    seed: int
    parameters: int
    checkpoint_path: str
    best_val_rmse: float
    test_block_rmse: float
    mean_rollout_rmse: float
    rmse_at_25: float
    rmse_at_50: float
    rmse_at_75: float
    rmse_at_100: float
    rollout_curve_path: str
    history_path: str
    # PFLOTRAN trajectories have 75 frames in this setup. With H=10,
    # the maximum rollout is 65. These fields record the actual final
    # evaluated forecast step so tables do not have to call it step 100.
    final_step: int = 0
    rmse_at_final: float = float("nan")


@dataclass
class FamilyResult:
    name: str
    label: str
    parameters: int
    run_records: List[RunRecord]
    curves: np.ndarray
    best_checkpoint: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch, device: torch.device):
    x, y, params, meta, pos, edge_index = batch
    return (
        x.to(device, non_blocking=True),
        y.to(device, non_blocking=True),
        params.to(device, non_blocking=True),
        meta,
        pos.to(device, non_blocking=True),
        edge_index.to(device, non_blocking=True),
    )


def rmse_from_mse(mse: float) -> float:
    return float(np.sqrt(max(float(mse), 0.0)))


@torch.no_grad()
def evaluate_block_rmse(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    total_sse = 0.0
    total_count = 0
    for batch in loader:
        x, y, *_ = move_batch_to_device(batch, device)
        pred = model(x)
        err = (pred - y).float()
        total_sse += float(err.pow(2).sum().item())
        total_count += int(err.numel())
    return rmse_from_mse(total_sse / max(total_count, 1))


def train_one_epoch(model: nn.Module, loader, optimizer, scaler, device: torch.device, cfg: TrainConfig) -> float:
    model.train()
    total_sse = 0.0
    total_count = 0
    use_amp = bool(cfg.amp and device.type == "cuda")

    for batch in loader:
        x, y, *_ = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(x)
            loss = F.mse_loss(pred, y)

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

        err = (pred.detach() - y).float()
        total_sse += float(err.pow(2).sum().item())
        total_count += int(err.numel())

    return rmse_from_mse(total_sse / max(total_count, 1))


@torch.no_grad()
def block_rollout(model: nn.Module, context: torch.Tensor, steps: int, history: int) -> torch.Tensor:
    """Return exactly `steps` autoregressive frames.

    context: (B,H,N,C), normalized.
    output:  (B,steps,N,C), normalized.
    """
    model.eval()
    current = context
    preds = []
    remaining = int(steps)
    while remaining > 0:
        block = model(current)
        take = min(remaining, block.shape[1])
        preds.append(block[:, :take])
        current = torch.cat([current, block], dim=1)[:, -history:]
        remaining -= take
    return torch.cat(preds, dim=1)


@torch.no_grad()
def evaluate_rollout_curves(
    model: nn.Module,
    info: Dict[str, object],
    device: torch.device,
    steps: int,
    history: int,
) -> np.ndarray:
    """Evaluate block-autoregressive curves over held-out scenarios.

    The returned array has length `steps`. Scenarios shorter than
    history + steps are skipped. If all test scenarios are skipped, a clear
    error is raised.
    """
    model.eval()
    stats_vars, _ = stats_from_jsonable(info["stats"])
    var_names = info["var_names"]
    curves = []
    for path in info["splits"]["test_paths"]:
        traj, _ = load_normalized_trajectory(path, var_names, stats_vars, normalize=bool(info.get("normalize", True)))
        if traj.shape[0] < history + steps:
            continue
        context = torch.from_numpy(traj[:history]).float().unsqueeze(0).to(device)
        target = torch.from_numpy(traj[history:history + steps]).float().unsqueeze(0).to(device)
        pred = block_rollout(model, context, steps=steps, history=history)
        err = (pred - target).float().squeeze(0)  # (steps,N,C)
        rmse_t = torch.sqrt(err.pow(2).mean(dim=(1, 2))).detach().cpu().numpy()
        curves.append(rmse_t)
    if not curves:
        raise RuntimeError("No test trajectories were long enough for the requested rollout.")
    return np.stack(curves, axis=0).mean(axis=0)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_json(path: Path) -> Dict[str, object]:
    with open(path, "r") as f:
        return json.load(f)


def run_single_seed(
    spec: ModelSpec,
    seed: int,
    train_loader,
    val_loader,
    test_loader,
    info: Dict[str, object],
    cfg: TrainConfig,
    device: torch.device,
    output_dir: Path,
    force: bool = False,
) -> RunRecord:
    model_dir = output_dir / "models"
    curve_dir = output_dir / "curves"
    hist_dir = output_dir / "histories"
    record_dir = output_dir / "records"
    for d in (model_dir, curve_dir, hist_dir, record_dir):
        d.mkdir(parents=True, exist_ok=True)

    safe = spec.name
    ckpt_path = model_dir / f"{safe}_seed{seed}_best.pt"
    curve_path = curve_dir / f"{safe}_seed{seed}_rollout_curve.npy"
    hist_path = hist_dir / f"{safe}_seed{seed}_history.csv"
    record_path = record_dir / f"{safe}_seed{seed}.json"

    if record_path.is_file() and ckpt_path.is_file() and curve_path.is_file() and not force:
        payload = load_json(record_path)
        return RunRecord(**payload)

    set_seed(seed)
    model = spec.factory().to(device)
    parameters = count_parameters(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.scheduler.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 0.01)
    else:
        scheduler = None

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp and device.type == "cuda"))

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    rows = []

    for epoch in range(1, cfg.epochs + 1):
        train_rmse = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg)
        val_rmse = evaluate_block_rmse(model, val_loader, device)
        lr_now = float(optimizer.param_groups[0]["lr"])
        rows.append({"epoch": epoch, "train_rmse": train_rmse, "val_rmse": val_rmse, "lr": lr_now})

        if scheduler is not None:
            scheduler.step()

        improved = val_rmse < best_val - 1e-8
        if improved:
            best_val = float(val_rmse)
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": spec.name,
                    "label": spec.label,
                    "seed": int(seed),
                    "parameters": int(parameters),
                    "best_val_rmse": float(best_val),
                    "epoch": int(epoch),
                    "config": asdict(cfg),
                    "info": info,
                },
                ckpt_path,
            )
        else:
            bad_epochs += 1

        print(
            f"[{spec.name} seed={seed}] epoch {epoch:03d} "
            f"train={train_rmse:.6f} val={val_rmse:.6f} lr={lr_now:.2e} "
            f"best={best_val:.6f}@{best_epoch}"
        )

        if bad_epochs >= cfg.early_stopping_patience:
            print(f"Early stopping {spec.name} seed={seed} after {epoch} epochs.")
            break

    pd.DataFrame(rows).to_csv(hist_path, index=False)

    # Load best checkpoint for test/rollout metrics.
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    test_block = evaluate_block_rmse(model, test_loader, device)
    curve = evaluate_rollout_curves(model, info, device, steps=cfg.rollout_steps, history=cfg.history)
    np.save(curve_path, curve.astype(np.float32))

    def at_or_nan(step: int) -> float:
        step = int(step)
        if 1 <= step <= len(curve):
            return float(curve[step - 1])
        return float("nan")

    final_step = int(len(curve))
    final_rmse = at_or_nan(final_step)

    record = RunRecord(
        model=spec.name,
        label=spec.label,
        seed=int(seed),
        parameters=int(parameters),
        checkpoint_path=str(ckpt_path),
        best_val_rmse=float(best_val),
        test_block_rmse=float(test_block),
        mean_rollout_rmse=float(curve.mean()),
        rmse_at_25=at_or_nan(25),
        rmse_at_50=at_or_nan(50),
        rmse_at_75=at_or_nan(75),
        rmse_at_100=at_or_nan(100),
        rollout_curve_path=str(curve_path),
        history_path=str(hist_path),
        final_step=final_step,
        rmse_at_final=final_rmse,
    )
    save_json(record_path, asdict(record))
    return record


def run_model_family(
    spec: ModelSpec,
    seeds: Sequence[int],
    train_loader,
    val_loader,
    test_loader,
    info: Dict[str, object],
    cfg: TrainConfig,
    device: torch.device,
    output_dir: Path,
    force: bool = False,
) -> FamilyResult:
    output_dir = Path(output_dir)
    family_dir = output_dir / "families"
    family_dir.mkdir(parents=True, exist_ok=True)
    family_path = family_dir / f"{spec.name}.json"
    curves_path = family_dir / f"{spec.name}_curves.npy"

    records = []
    curves = []
    for seed in seeds:
        rec = run_single_seed(spec, int(seed), train_loader, val_loader, test_loader, info, cfg, device, output_dir, force=force)
        records.append(rec)
        curves.append(np.load(rec.rollout_curve_path))

    curves_arr = np.stack(curves, axis=0).astype(np.float32)
    np.save(curves_path, curves_arr)
    best_idx = int(np.argmin([r.mean_rollout_rmse for r in records]))
    family = FamilyResult(
        name=spec.name,
        label=spec.label,
        parameters=spec.parameters,
        run_records=records,
        curves=curves_arr,
        best_checkpoint=records[best_idx].checkpoint_path,
    )
    save_json(
        family_path,
        {
            "name": family.name,
            "label": family.label,
            "parameters": family.parameters,
            "curves_path": str(curves_path),
            "best_checkpoint": family.best_checkpoint,
            "run_records": [asdict(r) for r in records],
        },
    )
    return family


def _record_from_payload(payload: Dict[str, object]) -> RunRecord:
    """Load a RunRecord while remaining compatible with older JSON files."""
    payload = dict(payload)
    if "final_step" not in payload:
        # Older records did not store the final evaluated rollout step. Infer
        # it from the saved curve when possible; otherwise leave it as zero.
        curve_path = payload.get("rollout_curve_path")
        if curve_path is not None and Path(str(curve_path)).is_file():
            curve = np.load(str(curve_path))
            payload["final_step"] = int(len(curve))
            payload["rmse_at_final"] = float(curve[-1]) if len(curve) > 0 else float("nan")
        else:
            payload["final_step"] = 0
            payload["rmse_at_final"] = float("nan")
    return RunRecord(**payload)


def load_family_result(model_name: str, output_dir: Path) -> Optional[FamilyResult]:
    family_path = Path(output_dir) / "families" / f"{model_name}.json"
    if not family_path.is_file():
        return None
    payload = load_json(family_path)
    curves = np.load(payload["curves_path"])
    records = [_record_from_payload(r) for r in payload["run_records"]]
    return FamilyResult(
        name=payload["name"],
        label=payload["label"],
        parameters=int(payload["parameters"]),
        run_records=records,
        curves=curves,
        best_checkpoint=payload["best_checkpoint"],
    )


def _nanmean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if np.all(np.isnan(x)):
        return float("nan")
    return float(np.nanmean(x))


def _nanstd(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if np.all(np.isnan(x)):
        return float("nan")
    return float(np.nanstd(x, ddof=0))


def aggregate_results(families: Sequence[FamilyResult]) -> pd.DataFrame:
    rows = []
    for fam in families:
        block = np.array([r.test_block_rmse for r in fam.run_records], dtype=float)
        mean_roll = np.array([r.mean_rollout_rmse for r in fam.run_records], dtype=float)
        s25 = np.array([r.rmse_at_25 for r in fam.run_records], dtype=float)
        s50 = np.array([r.rmse_at_50 for r in fam.run_records], dtype=float)
        s75 = np.array([r.rmse_at_75 for r in fam.run_records], dtype=float)
        s100 = np.array([r.rmse_at_100 for r in fam.run_records], dtype=float)
        sfinal = np.array([r.rmse_at_final for r in fam.run_records], dtype=float)
        final_steps = np.array([r.final_step for r in fam.run_records], dtype=int)
        final_step = int(final_steps[0]) if len(final_steps) and np.all(final_steps == final_steps[0]) else int(np.min(final_steps))

        rows.append({
            "model": fam.name,
            "Method": fam.label,
            "Parameters": int(fam.parameters),
            "Block test RMSE": float(block.mean()),
            "Std block test RMSE": float(block.std(ddof=0)),
            "Rollout RMSE": float(mean_roll.mean()),
            "Std rollout RMSE": float(mean_roll.std(ddof=0)),
            "Step 25 RMSE": _nanmean(s25),
            "Std step 25 RMSE": _nanstd(s25),
            "Step 50 RMSE": _nanmean(s50),
            "Std step 50 RMSE": _nanstd(s50),
            "Step 75 RMSE": _nanmean(s75),
            "Std step 75 RMSE": _nanstd(s75),
            "Step 100 RMSE": _nanmean(s100),
            "Std step 100 RMSE": _nanstd(s100),
            "Final step": final_step,
            "Final-step RMSE": _nanmean(sfinal),
            "Std final-step RMSE": _nanstd(sfinal),
            "Runs": len(fam.run_records),
            "selected_checkpoint": fam.best_checkpoint,
        })
    return pd.DataFrame(rows)


# SI uses its exact direct-horizon trainer, retained separately to avoid
# changing the PFLOTRAN checkpoint and rollout semantics above.
from tcwgno.si_benchmark.training_direct14 import (  # noqa: E402
    aggregate_direct_table as aggregate_si_direct_table,
    run_direct_family as run_si_direct_family,
)


def load_checkpoint_model(spec: ModelSpec, checkpoint_path: str, device: torch.device) -> nn.Module:
    model = spec.factory().to(device)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model
