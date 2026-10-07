from __future__ import annotations

import json
import math
import os
import random
import time
import fcntl
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import ALL_MODELS, CONFIG, TC_MODELS
from .data import AirfoilWindowDataset, iter_records, decode_state, load_meta, make_loader
from .registry import build_model_specs


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def loaders(seed: int):
    stats = json.loads((CONFIG.artifact_dir / "normalization.json").read_text())
    channel_indices = tuple(range(CONFIG.channels))
    train_ds = AirfoilWindowDataset(CONFIG.data_dir, "train", stats, CONFIG.history,
        CONFIG.forecast_horizon, CONFIG.windows_per_train_trajectory, seed, True,
        channel_indices=channel_indices)
    val_ds = AirfoilWindowDataset(CONFIG.data_dir, "valid", stats, CONFIG.history,
        CONFIG.forecast_horizon, CONFIG.windows_per_eval_trajectory, seed, False,
        channel_indices=channel_indices)
    test_ds = AirfoilWindowDataset(CONFIG.data_dir, "test", stats, CONFIG.history,
        CONFIG.forecast_horizon, CONFIG.windows_per_eval_trajectory, seed, False,
        channel_indices=channel_indices)
    return train_ds, make_loader(train_ds, CONFIG.batch_size), make_loader(val_ds, CONFIG.batch_size), make_loader(test_ds, CONFIG.batch_size)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval(); sse = 0.0; count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = model(x)
        err = pred.float() - y.float(); sse += err.square().sum().item(); count += err.numel()
    return math.sqrt(sse / max(count, 1))


@torch.no_grad()
def rollout_rmse(model, device, max_scenarios: int = 100) -> tuple[float, list[float], list[list[float]]]:
    model.eval(); meta = load_meta(CONFIG.data_dir)
    stats = json.loads((CONFIG.artifact_dir / "normalization.json").read_text())
    mean = np.asarray(stats["mean"], np.float32)[:CONFIG.channels]
    std = np.asarray(stats["std"], np.float32)[:CONFIG.channels]
    curves = []
    for idx, record in enumerate(iter_records(CONFIG.data_dir, "test")):
        if idx >= max_scenarios: break
        state = decode_state(record, meta)[..., :CONFIG.channels]
        state = (state - mean) / std
        context = torch.from_numpy(np.ascontiguousarray(state[:CONFIG.history])).unsqueeze(0).to(device)
        target = torch.from_numpy(np.ascontiguousarray(state[CONFIG.history:])).to(device)
        predictions = []
        while len(predictions) * CONFIG.forecast_horizon < CONFIG.rollout_steps:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                block = model(context).float()
            predictions.append(block.cpu())
            context = torch.cat([context, block.to(device)], 1)[:, -CONFIG.history:]
        pred = torch.cat(predictions, 1)[0, :CONFIG.rollout_steps]
        err = pred - target[:CONFIG.rollout_steps].cpu()
        curves.append(torch.sqrt(err.square().mean((1, 2))).numpy())
    scenario_curves = np.stack(curves)
    curve = scenario_curves.mean(0)
    return float(curve.mean()), curve.tolist(), scenario_curves.tolist()


def train_model(name: str, run: int, force: bool = False) -> dict:
    run_dir = CONFIG.output_dir / "runs" / f"run_{run:02d}"
    records_dir = run_dir / "records"; records_dir.mkdir(parents=True, exist_ok=True)
    histories_dir = run_dir / "histories"; histories_dir.mkdir(exist_ok=True)
    temp_dir = run_dir / "temporary_checkpoints"; temp_dir.mkdir(exist_ok=True)
    record_path = records_dir / f"{name}.json"
    if record_path.exists() and not force:
        return json.loads(record_path.read_text())
    specs, matched = build_model_specs(); spec = specs[name]
    run_seed = CONFIG.seed + run - 1
    seed_everything(run_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = spec.factory().to(device)
    train_ds, train_loader, val_loader, test_loader = loaders(run_seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=CONFIG.warmup_start_factor,
        end_factor=1.0,
        total_iters=CONFIG.warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG.epochs - CONFIG.warmup_epochs,
        eta_min=CONFIG.min_learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[CONFIG.warmup_epochs],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint = temp_dir / f"{name}_best.pt"
    best = float("inf"); best_epoch = 0; bad = 0; rows = []
    started = time.time()
    for epoch in range(1, CONFIG.epochs + 1):
        train_ds.set_epoch(epoch); model.train(); sse = 0.0; count = 0; epoch_start = time.time()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred = model(x); loss = F.mse_loss(pred, y)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.grad_clip_norm)
            scaler.step(optimizer); scaler.update()
            err = pred.detach().float() - y.float(); sse += err.square().sum().item(); count += err.numel()
        val = evaluate(model, val_loader, device); train = math.sqrt(sse / max(count, 1))
        rows.append({"run": run, "seed": run_seed, "epoch": epoch, "train_rmse": train, "val_rmse": val,
                     "seconds": time.time() - epoch_start, "lr": optimizer.param_groups[0]["lr"]})
        scheduler.step()
        if val < best:
            best, best_epoch, bad = val, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "model": name,
                        "parameters": spec.parameters, "matched": matched[name],
                        "epoch": epoch, "best_val_rmse": best, "config": asdict(CONFIG)}, checkpoint)
        else:
            bad += 1
        pd.DataFrame(rows).to_csv(histories_dir / f"{name}.csv", index=False)
        print(f"[{name} run={run:02d}] epoch={epoch:03d} train={train:.6f} val={val:.6f} best={best:.6f}@{best_epoch}", flush=True)
        if bad >= CONFIG.early_stopping_patience: break
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    test = evaluate(model, test_loader, device)
    roll, curve, scenario_curves = rollout_rmse(model, device)
    result = {"model": name, "label": spec.label, "run": run, "seed": run_seed,
              "parameters": spec.parameters,
              "matched": matched[name], "best_epoch": best_epoch, "best_val_rmse": best,
              "test_block_rmse": test, "mean_rollout_rmse": roll, "rollout_curve": curve,
              "rollout_scenario_curves": scenario_curves,
              "checkpoint": str(checkpoint), "total_hours": (time.time() - started) / 3600}
    record_path.write_text(json.dumps(result, indent=2) + "\n")
    del model; torch.cuda.empty_cache()
    return result


def run_suite(names) -> list[dict]:
    results = []
    for name in names:
        for run in range(1, CONFIG.runs_per_model + 1):
            results.append(train_model(name, run))
    return results


def _average_record(run_records: list[dict]) -> dict:
    """Create a plotting/ranking record while retaining run-level variability."""
    first = run_records[0]
    curves = np.asarray([record["rollout_curve"] for record in run_records], dtype=float)
    scalar_keys = ("best_epoch", "best_val_rmse", "test_block_rmse", "mean_rollout_rmse", "total_hours")
    averaged = {
        "model": first["model"], "label": first["label"],
        "parameters": first["parameters"], "matched": first["matched"],
        "run_count": len(run_records), "runs": run_records,
        "rollout_curve": curves.mean(axis=0).tolist(),
        "rollout_curve_std": curves.std(axis=0).tolist(),
    }
    for key in scalar_keys:
        values = np.asarray([record[key] for record in run_records], dtype=float)
        averaged[key] = float(values.mean())
        averaged[f"{key}_std"] = float(values.std())
    return averaged


def finalize_checkpoints() -> list[str]:
    CONFIG.output_dir.mkdir(parents=True, exist_ok=True)
    with (CONFIG.output_dir / ".finalize.lock").open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        return _finalize_checkpoints_locked()


def _finalize_checkpoints_locked() -> list[str]:
    records = {}
    for name in ALL_MODELS:
        run_records = []
        for run in range(1, CONFIG.runs_per_model + 1):
            path = CONFIG.output_dir / "runs" / f"run_{run:02d}" / "records" / f"{name}.json"
            if not path.exists():
                raise RuntimeError(f"Both suites and runs must finish first; missing {path}")
            run_records.append(json.loads(path.read_text()))
        records[name] = _average_record(run_records)
    averaged_dir = CONFIG.output_dir / "averaged" / "records"
    averaged_dir.mkdir(parents=True, exist_ok=True)
    for name, record in records.items():
        (averaged_dir / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    non_tc = sorted((v for k, v in records.items() if k not in TC_MODELS),
                    key=lambda x: x["mean_rollout_rmse"])
    keep = set(TC_MODELS) | {x["model"] for x in non_tc[:3]}
    final_dir = CONFIG.output_dir / "models"; final_dir.mkdir(exist_ok=True)
    for run in range(1, CONFIG.runs_per_model + 1):
        run_models_dir = final_dir / f"run_{run:02d}"; run_models_dir.mkdir(exist_ok=True)
        temp_dir = CONFIG.output_dir / "runs" / f"run_{run:02d}" / "temporary_checkpoints"
        for name in ALL_MODELS:
            path = temp_dir / f"{name}_best.pt"
            destination = run_models_dir / path.name
            if name in keep:
                if path.exists():
                    path.replace(destination)
                if not destination.exists():
                    raise RuntimeError(f"Missing retained checkpoint {destination}")
                records[name]["runs"][run - 1]["checkpoint"] = str(destination)
            else:
                if path.exists():
                    path.unlink()
                records[name]["runs"][run - 1]["checkpoint"] = ""
            source_record = CONFIG.output_dir / "runs" / f"run_{run:02d}" / "records" / f"{name}.json"
            source_record.write_text(json.dumps(records[name]["runs"][run - 1], indent=2) + "\n")
    for name, record in records.items():
        (averaged_dir / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    ranking = sorted(records.values(), key=lambda x: x["mean_rollout_rmse"])
    (CONFIG.output_dir / "ranking.json").write_text(json.dumps(ranking, indent=2) + "\n")
    (CONFIG.output_dir / "retained_models.json").write_text(json.dumps(sorted(keep), indent=2) + "\n")
    return sorted(keep)
