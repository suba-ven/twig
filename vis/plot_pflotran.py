from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from pflotran_block_dataset_3d_fixed import (
    denormalize_channel,
    load_normalized_trajectory,
    stats_from_jsonable,
)
from pflotran_h10_training import block_rollout, load_checkpoint_model


def plot_rollout_curves(
    families: Dict[str, object],
    model_order: Sequence[str],
    labels: Optional[Dict[str, str]] = None,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=160)
    for name in model_order:
        fam = families[name]
        curves = fam.curves
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        x = np.arange(1, mean.shape[0] + 1)
        label = labels.get(name, fam.label) if labels else fam.label
        ax.plot(x, mean, linewidth=2.2, label=label)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, linewidth=0)
    ax.set_xlabel("Block-autoregressive forecast step")
    ax.set_ylabel("RMSE")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8.5, ncol=2, frameon=True)
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig, ax


def write_latex_table(summary: pd.DataFrame, output_path: Path) -> None:
    compact = summary[["Method", "Parameters", "Rollout RMSE", "Std rollout RMSE", "Step 50 RMSE", "Std step 50 RMSE", "Step 100 RMSE", "Std step 100 RMSE"]].copy()
    compact["Rollout RMSE"] = [f"{m:.3f} $\\pm$ {s:.3f}" for m, s in zip(compact["Rollout RMSE"], compact["Std rollout RMSE"])]
    compact["Step 50 RMSE"] = [f"{m:.3f} $\\pm$ {s:.3f}" for m, s in zip(compact["Step 50 RMSE"], compact["Std step 50 RMSE"])]
    compact["Step 100 RMSE"] = [f"{m:.3f} $\\pm$ {s:.3f}" for m, s in zip(compact["Step 100 RMSE"], compact["Std step 100 RMSE"])]
    compact = compact[["Method", "Parameters", "Rollout RMSE", "Step 50 RMSE", "Step 100 RMSE"]]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(compact.to_latex(index=False, escape=False))


def _edge_segments_2d(pos: np.ndarray, edge_index: np.ndarray, axes=(0, 1)):
    if edge_index.shape[0] == 2:
        edges = edge_index.T
    else:
        edges = edge_index
    pairs = np.unique(np.sort(edges, axis=1), axis=0)
    xy = pos[:, axes]
    seg = np.stack([xy[pairs[:, 0]], xy[pairs[:, 1]]], axis=1)
    edge_x = np.column_stack([seg[:, 0, 0], seg[:, 1, 0], np.full(len(seg), np.nan)]).ravel()
    edge_y = np.column_stack([seg[:, 0, 1], seg[:, 1, 1], np.full(len(seg), np.nan)]).ravel()
    return edge_x, edge_y, xy


def make_qualitative_plotly(
    model,
    info: Dict[str, object],
    node_pos: np.ndarray,
    edge_index: np.ndarray,
    device: torch.device,
    output_html: Path,
    forecast_steps=(25, 50, 75, 100),
    scenario_local_index: int = 0,
    channel: int = 1,
    axes: Tuple[int, int] = (0, 1),
    title: Optional[str] = None,
    write_static: bool = True,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    history = int(info["history"])
    rollout_steps = max(forecast_steps)
    stats_vars, _ = stats_from_jsonable(info["stats"])
    var_names = info["var_names"]
    test_paths = info["splits"]["test_paths"]
    if not 0 <= scenario_local_index < len(test_paths):
        raise IndexError(f"scenario_local_index must be in [0,{len(test_paths)-1}]")

    traj, _ = load_normalized_trajectory(test_paths[scenario_local_index], var_names, stats_vars, normalize=bool(info.get("normalize", True)))
    if traj.shape[0] < history + rollout_steps:
        raise ValueError("Selected trajectory is too short for requested rollout.")

    context = torch.from_numpy(traj[:history]).float().unsqueeze(0).to(device)
    target = traj[history:history + rollout_steps]
    with torch.no_grad():
        pred = block_rollout(model, context, steps=rollout_steps, history=history).squeeze(0).cpu().numpy()

    # Convert selected channel back to raw physical scale.
    target_c = denormalize_channel(target[..., channel], info, channel)
    pred_c = denormalize_channel(pred[..., channel], info, channel)
    err_c = np.abs(pred_c - target_c)

    idxs = [s - 1 for s in forecast_steps]
    field_min = float(target_c.min())
    field_max = float(target_c.max())
    if np.isclose(field_min, field_max):
        field_max = field_min + 1e-8
    error_max = float(err_c[idxs].max())
    if np.isclose(error_max, 0.0):
        error_max = 1e-8

    edge_x, edge_y, xy = _edge_segments_2d(np.asarray(node_pos), np.asarray(edge_index), axes=axes)
    x_min, x_max = xy[:, 0].min(), xy[:, 0].max()
    y_min, y_max = xy[:, 1].min(), xy[:, 1].max()
    x_pad = 0.04 * max(x_max - x_min, 1e-8)
    y_pad = 0.04 * max(y_max - y_min, 1e-8)
    x_range = [x_min - x_pad, x_max + x_pad]
    y_range = [y_min - y_pad, y_max + y_pad]

    subplot_titles = [f"<b>Forecast step {s}</b>" for s in forecast_steps] + [""] * (2 * len(forecast_steps))
    fig = make_subplots(rows=3, cols=len(forecast_steps), subplot_titles=subplot_titles,
                        horizontal_spacing=0.025, vertical_spacing=0.050)

    for col, step in enumerate(forecast_steps, start=1):
        i = step - 1
        rows = [
            ("Target", target_c[i], "Viridis", field_min, field_max, col == len(forecast_steps),
             dict(title=dict(text=f"{var_names[channel]}<br>(raw scale)", side="right"), thickness=14, len=0.56, x=1.015, y=0.685, outlinewidth=0.5)),
            ("Prediction", pred_c[i], "Viridis", field_min, field_max, False, None),
            ("Absolute error", err_c[i], "Magma", 0.0, error_max, col == len(forecast_steps),
             dict(title=dict(text="Absolute<br>error", side="right"), thickness=14, len=0.25, x=1.015, y=0.165, outlinewidth=0.5)),
        ]
        for row, (label, values, colorscale, cmin, cmax, show_colorbar, colorbar) in enumerate(rows, start=1):
            fig.add_trace(
                go.Scatter(
                    x=edge_x, y=edge_y, mode="lines",
                    line=dict(color="rgba(0,0,0,0.38)", width=0.7),
                    hoverinfo="skip", showlegend=False,
                ),
                row=row, col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=xy[:, 0], y=xy[:, 1], mode="markers",
                    marker=dict(size=4.8, color=values, colorscale=colorscale, cmin=cmin, cmax=cmax,
                                showscale=show_colorbar, colorbar=colorbar, line=dict(width=0)),
                    customdata=np.column_stack([np.arange(len(values)), values]),
                    hovertemplate=(
                        "Node %{customdata[0]}<br>"
                        "x = %{x:.3f}<br>y = %{y:.3f}<br>"
                        f"{label} = %{{customdata[1]:.5g}}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row, col=col,
            )

    n_cols = len(forecast_steps)
    for row in range(1, 4):
        for col in range(1, n_cols + 1):
            axis_number = (row - 1) * n_cols + col
            x_axis_name = "x" if axis_number == 1 else f"x{axis_number}"
            fig.update_xaxes(visible=False, range=x_range, row=row, col=col)
            fig.update_yaxes(visible=False, range=y_range, scaleanchor=x_axis_name, scaleratio=1, row=row, col=col)

    for y, label in [(0.835, "Target"), (0.500, "Prediction"), (0.165, "Absolute error")]:
        fig.add_annotation(x=-0.045, y=y, xref="paper", yref="paper", text=f"<b>{label}</b>",
                           textangle=-90, showarrow=False, font=dict(size=16))

    if title is None:
        title = f"PFLOTRAN block-autoregressive rollout — {var_names[channel]}"
    fig.update_layout(
        template="simple_white", width=1450, height=890,
        margin=dict(l=95, r=125, t=78, b=26),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Arial, Helvetica, sans-serif", size=14, color="black"),
        title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center", y=0.99, yanchor="top", font=dict(size=18)),
    )

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_html, include_plotlyjs="cdn")

    if write_static:
        try:
            fig.write_image(output_html.with_suffix(".pdf"))
            fig.write_image(output_html.with_suffix(".svg"))
        except Exception as exc:
            print("Static export skipped; install kaleido/Chrome if needed. Details:", exc)
    return fig, {"target": target_c, "prediction": pred_c, "error": err_c}
