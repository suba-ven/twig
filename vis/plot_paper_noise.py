"""Reproduce the paper PFLOTRAN noise figure with TWIG in the legend.

Plot/style copied from cells 4 and 28 of
pflotran_3d/h10-benchmarks/08_pflotran_h10_final_paper_figures.ipynb.
Run: python vis/plot_paper_noise.py --output /results/figures
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import argparse
ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description='Replot frozen PFLOTRAN noise summaries.')
parser.add_argument('--output', type=Path, default=ROOT/'results/codeocean/figures')
FINAL_FIGURE_DIR = parser.parse_args().output
# ---------------------------------------------------------------------
# Shared paper style
# ---------------------------------------------------------------------
PAPER_WIDTH_IN = 7.2
PAPER_PANEL_HEIGHT_IN = 3.05

METHOD_ORDER = [
    "TWIG",
    "TWIG (K=3)",
    "GPS Transformer",
    "GAT",
    "MeshGraphNet",
    "Graph WNO",
    "GATv2",
    "RNN-GNN Fusion",
    "RNN",
    "Graph FNO",
]

COMPETITIVE_METHODS = [
    "TWIG",
    "TWIG (K=3)",
    "GPS Transformer",
    "GAT",
    "MeshGraphNet",
    "Graph WNO",
    "GATv2",
]

NOISE_METHODS = [
    "TWIG",
    "Graph WNO",
    "MeshGraphNet",
    "GPS Transformer",
    "GAT",
]

METHOD_LABEL_RENAMES = {
    "SA TWIG + SwiGLU": "TWIG",
    "SA TWIG + SwiGLU (K=5)": "TWIG",
    "SA TWIG + SwiGLU (K=3)": "TWIG (K=3)",
    "Plain Graph WNO": "Graph WNO",
}

# Stable colors taken from Matplotlib's built-in tab10 palette.
tab10 = plt.get_cmap("tab10")
METHOD_COLORS = {
    method: tab10(index % 10)
    for index, method in enumerate(METHOD_ORDER)
}

METHOD_MARKERS = {
    "TWIG": "o",
    "TWIG (K=3)": "s",
    "GPS Transformer": "D",
    "GAT": "v",
    "MeshGraphNet": "^",
    "Graph WNO": "P",
    "GATv2": "X",
    "RNN-GNN Fusion": "<",
    "RNN": ">",
    "Graph FNO": "*",
}

METHOD_LINESTYLES = {
    "TWIG": "-",
    "TWIG (K=3)": "--",
    "GPS Transformer": ":",
    "GAT": (0, (5, 2)),
    "MeshGraphNet": "-.",
    "Graph WNO": (0, (3, 1, 1, 1)),
    "GATv2": (0, (1, 1)),
    "RNN-GNN Fusion": (0, (6, 2, 1, 2)),
    "RNN": (0, (2, 2)),
    "Graph FNO": "-",
}

def set_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 8.3,
        "axes.labelsize": 8.7,
        "axes.titlesize": 9.0,
        "legend.fontsize": 6.9,
        "xtick.labelsize": 7.7,
        "ytick.labelsize": 7.7,
        "axes.linewidth": 0.75,
        "lines.linewidth": 1.55,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })

def save_matplotlib_figure(
    figure,
    stem,
    *,
    directory=FINAL_FIGURE_DIR,
):
    directory = Path(directory)
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_path = directory / f"{stem}.pdf"
    png_path = directory / f"{stem}.png"

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )
    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    print("Saved:", pdf_path)
    print("Saved:", png_path)

    return pdf_path, png_path

def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(
        axis="y",
        linewidth=0.42,
        alpha=0.25,
    )

set_paper_style()

NOISE_SUMMARY_PATH = ROOT / 'results/paper/pflotran_noise_summary.csv'
noise_summary = pd.read_csv(NOISE_SUMMARY_PATH)
noise_summary['Method'] = noise_summary['Method'].replace(METHOD_LABEL_RENAMES)
assert all((noise_summary['Method'] == m).sum() == 5 for m in NOISE_METHODS)

# ---------------------------------------------------------------------
# Paper-ready two-panel noise robustness figure
# ---------------------------------------------------------------------
noise_panel_specs = [
    {
        "mean": "mean_rollout_rmse",
        "std": "std_rollout_rmse",
        "ylabel": "Mean rollout RMSE",
        "panel": "(a)",
    },
    {
        "mean": "mean_step60_rmse",
        "std": "std_step60_rmse",
        "ylabel": "Final Timestep RMSE",
        "panel": "(b)",
    },
]

fig, axes = plt.subplots(
    1,
    2,
    figsize=(
        PAPER_WIDTH_IN,
        PAPER_PANEL_HEIGHT_IN,
    ),
    sharex=True,
)

for ax, panel_spec in zip(
    axes,
    noise_panel_specs,
):
    for method in NOISE_METHODS:
        frame = (
            noise_summary.loc[
                noise_summary[
                    "Method"
                ] == method
            ]
            .sort_values(
                "noise_percent_of_channel_std"
            )
        )

        if frame.empty:
            warnings.warn(
                f"No noise results found for {method}."
            )
            continue

        x = frame[
            "noise_percent_of_channel_std"
        ].to_numpy(
            dtype=float
        )
        y = frame[
            panel_spec["mean"]
        ].to_numpy(
            dtype=float
        )
        yerr = frame[
            panel_spec["std"]
        ].to_numpy(
            dtype=float
        )

        linewidth = (
            2.15
            if method == "TWIG"
            else 1.40
        )

        line, = ax.plot(
            x,
            y,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=linewidth,
            markersize=3.8,
            markeredgewidth=0.45,
            label=method,
            zorder=(
                5
                if method == "TWIG"
                else 2
            ),
        )

        ax.fill_between(
            x,
            np.maximum(
                y - yerr,
                0.0,
            ),
            y + yerr,
            color=line.get_color(),
            alpha=0.085,
            linewidth=0.0,
            zorder=0,
        )

    ax.set_ylabel(
        panel_spec["ylabel"]
    )
    ax.set_xlim(
        -0.25,
        10.25,
    )
    ax.set_xticks(
        [0, 1, 2.5, 5, 10]
    )
    ax.set_xticklabels(
        ["0", "1", "2.5", "5", "10"]
    )
    ax.yaxis.set_major_formatter(
        mticker.FormatStrFormatter(
            "%.03f"
        )
    )

    clean_axis(ax)

    ax.text(
        0.02,
        0.97,
        panel_spec["panel"],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=8.8,
    )

fig.supxlabel(
    "Input noise (% of channel standard deviation)",
    y=0.045,
    fontsize=8.7,
)

noise_handles = [
    Line2D(
        [0],
        [0],
        color=METHOD_COLORS[method],
        marker=METHOD_MARKERS[method],
        linestyle=METHOD_LINESTYLES[method],
        linewidth=(
            2.1
            if method == "TWIG"
            else 1.4
        ),
        markersize=3.5,
        label=method,
    )
    for method in NOISE_METHODS
]

fig.legend(
    handles=noise_handles,
    labels=NOISE_METHODS,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.005),
    ncol=5,
    frameon=False,
    columnspacing=0.95,
    handlelength=2.1,
    handletextpad=0.38,
)

fig.subplots_adjust(
    left=0.09,
    right=0.995,
    bottom=0.20,
    top=0.80,
    wspace=0.28,
)

noise_pdf, noise_png = (
    save_matplotlib_figure(
        fig,
        "pflotran-noise-robustness-twig",
    )
)

plt.close(fig)
