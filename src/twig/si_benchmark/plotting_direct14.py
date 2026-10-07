from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .training_direct14 import DirectFamilyResult


def _mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    if values.shape[0] < 2:
        return mean, np.zeros_like(mean)
    return mean, values.std(axis=0, ddof=1)


def plot_direct14_rollout(
    families: dict[str, DirectFamilyResult],
    names: list[str],
    labels: dict[str, str],
    output_dir: Path,
    stem: str = "direct14_block_rollout_mean_std",
) -> tuple[Path, Path]:
    if not names:
        raise ValueError("names must contain at least one model.")

    lengths = {
        int(families[name].curves.shape[1])
        for name in names
    }
    if len(lengths) != 1:
        raise ValueError(
            "All families must have the same rollout length; got "
            f"{sorted(lengths)}."
        )

    rollout_steps = lengths.pop()
    steps = np.arange(1, rollout_steps + 1)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    for name in names:
        curves = np.asarray(families[name].curves)
        mean, std = _mean_std(curves)

        line = ax.plot(
            steps,
            mean,
            linewidth=2.0,
            label=f"{labels[name]} ($n={curves.shape[0]}$)",
        )[0]

        ax.fill_between(
            steps,
            mean - std,
            mean + std,
            color=line.get_color(),
            alpha=0.16,
            linewidth=0,
        )

    ax.set_xlim(1, rollout_steps)
    tick_candidates = [1, 25, 50, 75, 100]
    ax.set_xticks(
        [tick for tick in tick_candidates if tick <= rollout_steps]
    )
    ax.set_xlabel("Block-autoregressive forecast step")
    ax.set_ylabel("RMSE")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, frameon=True, ncol=2)

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"

    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    return png, pdf


def plot_direct14_loss_history(
    family: DirectFamilyResult,
    output_dir: Path,
    label: str,
    stem: str | None = None,
) -> tuple[Path, Path] | None:
    train = np.asarray(family.train_rmse_histories)
    val = np.asarray(family.val_rmse14_histories)

    if (
        train.ndim != 2
        or val.ndim != 2
        or train.shape[0] == 0
        or val.shape[0] == 0
    ):
        return None

    common_length = min(train.shape[1], val.shape[1])
    train = train[:, :common_length]
    val = val[:, :common_length]

    train_mean, train_std = _mean_std(train)
    val_mean, val_std = _mean_std(val)
    epochs = np.arange(1, common_length + 1)

    fig, ax = plt.subplots(figsize=(5.0, 3.2))

    train_line = ax.plot(
        epochs,
        train_mean,
        linewidth=1.8,
        linestyle="--",
        label=f"Train direct-14 RMSE ($n={train.shape[0]}$)",
    )[0]
    ax.fill_between(
        epochs,
        train_mean - train_std,
        train_mean + train_std,
        color=train_line.get_color(),
        alpha=0.16,
        linewidth=0,
    )

    val_line = ax.plot(
        epochs,
        val_mean,
        color="black",
        linewidth=1.8,
        label=f"Validation direct-14 RMSE ($n={val.shape[0]}$)",
    )[0]
    ax.fill_between(
        epochs,
        val_mean - val_std,
        val_mean + val_std,
        color=val_line.get_color(),
        alpha=0.13,
        linewidth=0,
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_title(label)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, frameon=True)

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or f"{family.name.lower().replace('-', '_')}_loss_history"
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"

    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    return png, pdf
