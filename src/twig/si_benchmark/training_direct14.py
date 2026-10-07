from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data_direct14 import PreparedDirect14Data
from .direct14_config import Direct14Config
from .direct14_registry import DirectModelSpec, count_parameters


@dataclass
class DirectRunResult:
    record: dict[str, Any]
    rollout_curve: np.ndarray
    state_dict: dict[str, torch.Tensor]


@dataclass
class DirectFamilyResult:
    name: str
    run_records: list[dict[str, Any]]
    curves: np.ndarray
    best_checkpoint: Path
    statistics_path: Path
    train_rmse_histories: np.ndarray
    val_rmse14_histories: np.ndarray


def set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def make_direct_loaders(
    data: PreparedDirect14Data,
    cfg: Direct14Config,
    seed: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    loader_kwargs = {
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }

    train = DataLoader(
        data.train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        **loader_kwargs,
    )
    val = DataLoader(
        data.val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test = DataLoader(
        data.test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train, val, test


def _assert_direct_shape(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    model_name: str,
) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"{model_name} produced {tuple(prediction.shape)}, but the "
            f"direct target is {tuple(target.shape)}. Direct models must "
            "return [batch, forecast_horizon, nodes, channels]."
        )


def rmse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Scalar RMSE objective over all samples, forecast frames, and nodes."""
    return torch.sqrt(F.mse_loss(prediction, target) + epsilon)


@torch.inference_mode()
def direct14_rmse(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    model_name: str,
) -> tuple[float, np.ndarray]:
    """Global direct RMSE and per-forecast-frame RMSE for one loader."""
    model.eval()

    sse = None
    count = 0

    for context, target in loader:
        context = context.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        prediction = model(context)
        _assert_direct_shape(
            prediction,
            target,
            model_name=model_name,
        )

        error = (prediction - target).double().square()

        if sse is None:
            sse = torch.zeros(
                target.shape[1],
                device=device,
                dtype=torch.float64,
            )

        sse += error.sum(dim=(0, 2, 3))
        count += (
            error.shape[0]
            * error.shape[2]
            * error.shape[3]
        )

    if sse is None or count == 0:
        raise RuntimeError("The evaluation loader produced no samples.")

    daily_rmse = torch.sqrt(sse / count)
    global_rmse = torch.sqrt(sse.sum() / (count * daily_rmse.numel()))

    return float(global_rmse.item()), daily_rmse.cpu().numpy()


@torch.inference_mode()
def block_rollout(
    model: torch.nn.Module,
    context: torch.Tensor,
    steps: int,
    *,
    history: int,
    forecast_horizon: int,
    model_name: str = "model",
) -> torch.Tensor:
    """Autoregressively chain direct 14-frame blocks to a requested horizon."""
    if context.ndim != 4:
        raise ValueError(
            f"Expected context [B,H,N,C], got {tuple(context.shape)}."
        )
    if context.shape[1] != history:
        raise ValueError(
            f"Expected H={history}, got context shape {tuple(context.shape)}."
        )

    remaining = int(steps)
    if remaining < 1:
        raise ValueError("steps must be positive.")

    rolling = context
    outputs: list[torch.Tensor] = []

    while remaining > 0:
        direct_block = model(rolling)

        expected = (
            rolling.shape[0],
            forecast_horizon,
            rolling.shape[2],
            rolling.shape[3],
        )
        if tuple(direct_block.shape) != expected:
            raise ValueError(
                f"{model_name} produced direct block "
                f"{tuple(direct_block.shape)}, expected {expected}."
            )

        take = min(remaining, forecast_horizon)
        outputs.append(direct_block[:, :take])

        # Advance the H-frame context using the entire direct prediction
        # block, then retain the latest H states.
        rolling = torch.cat([rolling, direct_block], dim=1)[:, -history:]
        remaining -= take

    return torch.cat(outputs, dim=1)


@torch.inference_mode()
def scenario_block_rollout_curve(
    model: torch.nn.Module,
    data: PreparedDirect14Data,
    cfg: Direct14Config,
    device: torch.device,
    *,
    model_name: str,
) -> np.ndarray:
    model.eval()

    rollout_steps = data.effective_rollout_steps(
        cfg.rollout_steps,
        cfg.history,
    )
    if rollout_steps <= 0:
        raise ValueError("The dataset has no frames available after the history window.")

    sse = torch.zeros(
        rollout_steps,
        device=device,
        dtype=torch.float64,
    )
    count = torch.zeros(
        rollout_steps,
        device=device,
        dtype=torch.float64,
    )

    for scenario_id in data.rollout_scenario_ids:
        scenario = data.infected_scenarios[scenario_id].to(
            device,
            non_blocking=True,
        )

        required_frames = cfg.history + rollout_steps
        if scenario.shape[0] < required_frames:
            raise ValueError(
                f"Scenario {scenario_id} has {scenario.shape[0]} frames, "
                f"but H={cfg.history} and rollout={rollout_steps} "
                f"require {required_frames}."
            )

        context = scenario[:cfg.history].unsqueeze(0)
        target = scenario[
            cfg.history : cfg.history + rollout_steps
        ].unsqueeze(0)

        prediction = block_rollout(
            model,
            context,
            rollout_steps,
            history=cfg.history,
            forecast_horizon=cfg.forecast_horizon,
            model_name=model_name,
        )

        error = (prediction - target).double().square()
        sse += error.sum(dim=(0, 2, 3))
        count += (
            error.shape[0]
            * error.shape[2]
            * error.shape[3]
        )

    return torch.sqrt(sse / count).cpu().numpy()


def _epoch_message(
    run_index: int,
    epoch: int,
    train_rmse: float,
    val_rmse: float,
    best_val_rmse: float,
    epoch_seconds: float,
    stale_epochs: int,
) -> str:
    return (
        f"Run {run_index} | epoch {epoch} | "
        f"train direct-14 RMSE={train_rmse:.6f} | "
        f"val direct-14 RMSE={val_rmse:.6f} | "
        f"best={best_val_rmse:.6f} | "
        f"stale={stale_epochs} | "
        f"time={epoch_seconds:.1f}s"
    )


def train_direct_one_run(
    spec: DirectModelSpec,
    seed: int,
    run_index: int,
    data: PreparedDirect14Data,
    cfg: Direct14Config,
    device: torch.device,
) -> DirectRunResult:
    set_seed(seed, device)

    model = spec.factory().to(device)
    train_loader, val_loader, test_loader = make_direct_loaders(
        data,
        cfg,
        seed,
        device,
    )

    # Matches the reference benchmark's optimizer family: Adam with an
    # RMSE objective and no learning-rate scheduler.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    train_history: list[float] = []
    val_history: list[float] = []

    started = time.time()

    epoch_progress = tqdm(
        range(1, cfg.epochs + 1),
        desc=f"Run {run_index} | {spec.name}",
        unit="epoch",
        dynamic_ncols=True,
        leave=True,
    )

    for epoch in epoch_progress:
        epoch_started = time.perf_counter()
        model.train()

        sse = torch.zeros((), device=device, dtype=torch.float64)
        count = 0

        for context, target in train_loader:
            context = context.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(context)
            _assert_direct_shape(
                prediction,
                target,
                model_name=spec.name,
            )

            data_loss = rmse_loss(prediction, target)
            loss = data_loss
            if spec.regularizer is not None:
                loss = loss + spec.regularizer(model)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss for {spec.name}, seed {seed}, "
                    f"epoch {epoch}."
                )

            loss.backward()
            optimizer.step()

            error = prediction.detach() - target
            sse += error.double().square().sum()
            count += error.numel()

        train_rmse = float(
            torch.sqrt(sse / max(count, 1)).item()
        )
        val_rmse, _ = direct14_rmse(
            model,
            val_loader,
            device,
            model_name=spec.name,
        )

        train_history.append(train_rmse)
        val_history.append(val_rmse)

        if val_rmse < best_val:
            best_val = val_rmse
            best_epoch = epoch
            best_state = cpu_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_seconds = time.perf_counter() - epoch_started
        epoch_progress.set_postfix_str(
            f"train={train_rmse:.5f}, "
            f"val={val_rmse:.5f}, "
            f"best={best_val:.5f}, "
            f"epoch={epoch_seconds:.1f}s"
        )
        epoch_progress.write(
            _epoch_message(
                run_index=run_index,
                epoch=epoch,
                train_rmse=train_rmse,
                val_rmse=val_rmse,
                best_val_rmse=best_val,
                epoch_seconds=epoch_seconds,
                stale_epochs=epochs_without_improvement,
            )
        )

        if (
            cfg.early_stopping_patience > 0
            and epochs_without_improvement
            >= cfg.early_stopping_patience
        ):
            epoch_progress.write(
                f"Early stopping {spec.name}, run {run_index}, at "
                f"epoch {epoch}: no validation improvement for "
                f"{cfg.early_stopping_patience} epochs."
            )
            break

    if best_state is None:
        raise RuntimeError(
            f"No valid checkpoint was found for {spec.name}, seed {seed}."
        )

    model.load_state_dict(best_state, strict=True)

    test_rmse, test_daily_rmse = direct14_rmse(
        model,
        test_loader,
        device,
        model_name=spec.name,
    )
    rollout_curve = scenario_block_rollout_curve(
        model,
        data,
        cfg,
        device,
        model_name=spec.name,
    )

    record: dict[str, Any] = {
        "model": spec.name,
        "seed": int(seed),
        "run_index": int(run_index),
        "parameters": int(count_parameters(model)),
        "best_epoch": int(best_epoch),
        "best_val_direct14_rmse": float(best_val),
        "test_direct14_rmse": float(test_rmse),
        "test_direct14_daily_rmse": test_daily_rmse.tolist(),
        "mean_100step_block_rollout_rmse": float(rollout_curve.mean()),
        "rmse_at_25": float(rollout_curve[24]),
        "rmse_at_50": float(rollout_curve[49]),
        "rmse_at_75": float(rollout_curve[74]),
        "rmse_at_100": float(rollout_curve[99]),
        "elapsed_seconds": float(time.time() - started),
        "train_rmse_history": train_history,
        "val_rmse14_history": val_history,
        **(spec.metadata or {}),
    }

    state_dict = best_state

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return DirectRunResult(
        record=record,
        rollout_curve=rollout_curve,
        state_dict=state_dict,
    )


def _slug(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def _json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def _validation_score(record: dict[str, Any]) -> float:
    value = float(record.get("best_val_direct14_rmse", float("inf")))
    return value if np.isfinite(value) else float("inf")


def _stack_histories(
    run_records: list[dict[str, Any]],
    key: str,
) -> np.ndarray:
    histories: list[np.ndarray] = []
    expected_length: int | None = None

    for record in run_records:
        values = record.get(key)
        if values is None:
            continue

        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size == 0 or not np.isfinite(array).all():
            continue

        if expected_length is None:
            expected_length = int(array.size)

        # Early stopping can make history lengths differ. Only keep common
        # histories for aggregate loss plots, truncating all to the shortest
        # length in a second pass below.
        histories.append(array)

    if not histories:
        return np.empty((0, 0), dtype=float)

    common_length = min(array.size for array in histories)
    return np.stack(
        [array[:common_length] for array in histories],
        axis=0,
    )


def _mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim == 0:
        values = values.reshape(1)

    mean = values.mean(axis=0)
    if values.shape[0] < 2:
        return mean, np.zeros_like(mean)

    return mean, values.std(axis=0, ddof=1)


def save_direct_family_result(
    name: str,
    run_records: list[dict[str, Any]],
    curves: list[np.ndarray],
    best_state_dict: dict[str, torch.Tensor],
    selected_record: dict[str, Any],
    output_dir: Path,
) -> DirectFamilyResult:
    slug = _slug(name)
    stats_dir = output_dir / "statistics"
    models_dir = output_dir / "models"
    stats_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    curve_array = np.stack(curves, axis=0)
    curve_mean, curve_std = _mean_std(curve_array)

    train_histories = _stack_histories(
        run_records,
        "train_rmse_history",
    )
    val_histories = _stack_histories(
        run_records,
        "val_rmse14_history",
    )

    np.save(
        stats_dir / f"{slug}_rollout_curves.npy",
        curve_array,
    )
    np.save(
        stats_dir / f"{slug}_rollout_mean_std.npy",
        np.stack([curve_mean, curve_std], axis=0),
    )
    np.save(
        stats_dir / f"{slug}_train_rmse_histories.npy",
        train_histories,
    )
    np.save(
        stats_dir / f"{slug}_val_rmse14_histories.npy",
        val_histories,
    )

    if train_histories.size > 0 and val_histories.size > 0:
        train_mean, train_std = _mean_std(train_histories)
        val_mean, val_std = _mean_std(val_histories)
        np.savez(
            stats_dir / f"{slug}_loss_history_mean_std.npz",
            epochs=np.arange(1, train_mean.size + 1),
            train_rmse_mean=train_mean,
            train_rmse_std=train_std,
            val_rmse14_mean=val_mean,
            val_rmse14_std=val_std,
            n_train_runs=np.array(train_histories.shape[0]),
            n_val_runs=np.array(val_histories.shape[0]),
        )

    pd.DataFrame(
        {
            "step": np.arange(1, curve_mean.size + 1),
            "mean_rmse": curve_mean,
            "std_rmse": curve_std,
        }
    ).to_csv(
        stats_dir / f"{slug}_rollout_mean_std.csv",
        index=False,
    )
    pd.DataFrame(run_records).to_csv(
        stats_dir / f"{slug}_run_metrics.csv",
        index=False,
    )
    _json_dump(
        run_records,
        stats_dir / f"{slug}_run_metrics.json",
    )

    checkpoint = {
        "model_name": name,
        "selected_seed": selected_record["seed"],
        "selected_best_epoch": selected_record["best_epoch"],
        "selected_best_val_direct14_rmse": selected_record[
            "best_val_direct14_rmse"
        ],
        "model_state_dict": best_state_dict,
    }
    best_checkpoint = models_dir / f"{slug}_best.pt"
    torch.save(checkpoint, best_checkpoint)

    return DirectFamilyResult(
        name=name,
        run_records=run_records,
        curves=curve_array,
        best_checkpoint=best_checkpoint,
        statistics_path=(
            stats_dir / f"{slug}_run_metrics.json"
        ),
        train_rmse_histories=train_histories,
        val_rmse14_histories=val_histories,
    )


def load_direct_family_result(
    name: str,
    output_dir: Path,
) -> DirectFamilyResult | None:
    slug = _slug(name)
    stats_dir = output_dir / "statistics"

    records_path = stats_dir / f"{slug}_run_metrics.json"
    curves_path = stats_dir / f"{slug}_rollout_curves.npy"
    train_path = stats_dir / f"{slug}_train_rmse_histories.npy"
    val_path = stats_dir / f"{slug}_val_rmse14_histories.npy"
    checkpoint = output_dir / "models" / f"{slug}_best.pt"

    if not (
        records_path.is_file()
        and curves_path.is_file()
        and checkpoint.is_file()
    ):
        return None

    with records_path.open() as handle:
        records = json.load(handle)

    curves = np.load(curves_path)
    train_histories = (
        np.load(train_path)
        if train_path.is_file()
        else _stack_histories(records, "train_rmse_history")
    )
    val_histories = (
        np.load(val_path)
        if val_path.is_file()
        else _stack_histories(records, "val_rmse14_history")
    )

    return DirectFamilyResult(
        name=name,
        run_records=records,
        curves=curves,
        best_checkpoint=checkpoint,
        statistics_path=records_path,
        train_rmse_histories=train_histories,
        val_rmse14_histories=val_histories,
    )


def _cached_family_matches(
    family: DirectFamilyResult,
    seeds: tuple[int, ...],
) -> bool:
    observed_seeds = [
        int(record.get("seed", -1))
        for record in family.run_records
    ]

    return (
        len(family.run_records) == len(seeds)
        and family.curves.shape[0] == len(seeds)
        and observed_seeds == [int(seed) for seed in seeds]
    )


def run_direct_family(
    spec: DirectModelSpec,
    seeds: tuple[int, ...],
    data: PreparedDirect14Data,
    cfg: Direct14Config,
    device: torch.device,
    output_dir: Path,
    *,
    force: bool = False,
) -> DirectFamilyResult:
    if not force:
        cached = load_direct_family_result(spec.name, output_dir)
        if cached is not None and _cached_family_matches(
            cached,
            seeds,
        ):
            return cached

    run_records: list[dict[str, Any]] = []
    curves: list[np.ndarray] = []
    selected_record: dict[str, Any] | None = None
    selected_state: dict[str, torch.Tensor] | None = None

    for run_index, seed in enumerate(seeds, start=1):
        result = train_direct_one_run(
            spec=spec,
            seed=seed,
            run_index=run_index,
            data=data,
            cfg=cfg,
            device=device,
        )

        run_records.append(result.record)
        curves.append(result.rollout_curve)

        if (
            selected_record is None
            or _validation_score(result.record)
            < _validation_score(selected_record)
        ):
            selected_record = result.record
            selected_state = result.state_dict
        else:
            del result.state_dict

        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    if selected_record is None or selected_state is None:
        raise RuntimeError(f"No result was produced for {spec.name}.")

    family = save_direct_family_result(
        name=spec.name,
        run_records=run_records,
        curves=curves,
        best_state_dict=selected_state,
        selected_record=selected_record,
        output_dir=output_dir,
    )

    del selected_state
    gc.collect()
    return family


def aggregate_direct_table(
    families: list[DirectFamilyResult],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for family in families:
        table = pd.DataFrame(family.run_records)

        def mean_std(column: str) -> tuple[float, float]:
            values = table[column].to_numpy(dtype=float)
            return (
                float(values.mean()),
                (
                    float(values.std(ddof=1))
                    if values.size > 1
                    else 0.0
                ),
            )

        direct14_mean, direct14_std = mean_std("test_direct14_rmse")
        rollout_mean, rollout_std = mean_std(
            "mean_100step_block_rollout_rmse"
        )
        rmse25, std25 = mean_std("rmse_at_25")
        rmse50, std50 = mean_std("rmse_at_50")
        rmse75, std75 = mean_std("rmse_at_75")
        rmse100, std100 = mean_std("rmse_at_100")

        rows.append(
            {
                "model": family.name,
                "parameters": int(table["parameters"].iloc[0]),
                "test_direct14_rmse": direct14_mean,
                "std_test_direct14_rmse": direct14_std,
                "mean_100step_block_rollout_rmse": rollout_mean,
                "std_100step_block_rollout_rmse": rollout_std,
                "rmse_at_25": rmse25,
                "std_rmse_at_25": std25,
                "rmse_at_50": rmse50,
                "std_rmse_at_50": std50,
                "rmse_at_75": rmse75,
                "std_rmse_at_75": std75,
                "rmse_at_100": rmse100,
                "std_rmse_at_100": std100,
                "n_runs": int(family.curves.shape[0]),
                "selected_checkpoint": str(family.best_checkpoint),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("mean_100step_block_rollout_rmse")
        .reset_index(drop=True)
    )
