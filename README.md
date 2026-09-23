# TWIG: three-dataset reproducibility package

Code and selected experiment results for the temporal wavelet graph operator
(labelled **TC-WGNO** in the saved experiments), covering **PFLOTRAN,
SI diffusion, and Airfoil**. The default training commands now follow the final
nine-model comparisons.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

Run commands from the repository root. PyTorch Geometric must match your
PyTorch/CUDA installation. Airfoil uses TFRecord files and the `tfrecord` package.

## Selected experiments

[configs/paper_selections.json](configs/paper_selections.json) records all 27
model/dataset selections, parameter counts, source provenance, selected
field checkpoints, and Airfoil's per-run training settings.

| Dataset | Context / direct forecast | Rollout | TWIG |
|---|---|---|---|---|
| PFLOTRAN | 10 / 10 | 60 | K=5, S=4, depth 10, width 120, SwiGLU; 1,000,620 parameters |
| SI diffusion | 14 / 14 | 100 | K=5, S=4, depth 8, width 34, SwiGLU; 70,233 parameters |
| Original Airfoil200 | 20 / 20 | 180 | K=5, S=12, depth 10, width 248, SwiGLU; 9,875,176 parameters |

Airfoil GPS uses depth 6, heads 6,
width 396 and the three runs with the **largest saved mean rollout RMSE**
among seven completed runs. The selection is frozen; new runs are not reranked.
Airfoil Graph FNO uses depth 14, width 64, **169 modes**, overriding the inherited
128-mode checkpoint config. Use the original contiguous 200-state dataset:
stride-2 and stride-3 datasets are different experiments.

PFLOTRAN uses a 1M target capacity, up to 100 epochs, AdamW at 5e-4 with cosine
decay to 1e-6, weight decay 1e-5, and patience 10.
SI uses 30 epochs, Adam at 3e-4, weight decay 1e-5, and patience 10.
Airfoil uses 100 epochs with the saved per-model batch sizes (4 or 8), warmup
and cosine schedule. Its runner restores each selected run's checkpoint config.

## Replot the reported results

The small frozen three-run arrays are included in `results/paper/`. No training
data, checkpoints, or cluster paths are needed:

```bash
python vis/plot_selected_results.py
# SI notebook 45 / table uses population SD; extended-data plots use sample SD:
python vis/plot_selected_results.py --dataset si_diffusion --ddof 0
```

This writes PDF/PNG curves and a CSV summary under `figures/selected/`.
`--ddof 1` is the default and matches the extended-data figures. Original
exports are available as `figures/<dataset>/selected_nine_models.{pdf,png}`.
Paths in the selection manifest describe the original experiment provenance;
the plotting command reads the bundled arrays.

## Data and training

The existing downloader supplies PFLOTRAN and the shortened SI data package:

```bash
bash scripts/download_data.sh
python scripts/run_experiments/run_pflotran_benchmarks.py data/pflotran/raw/*.h5
python scripts/run_experiments/run_si_benchmarks.py --data /path/to/SI_equation_dataset.npy
```

The final SI 100-step rollout needs at least 114 states (14 context + 100
forecast). The first-100 download supports only 86 forecast steps and cannot
reproduce the final 100-step comparison. Supply the full original SI array for
the reported horizon. Edges and coordinates default to the downloaded files.

Airfoil data is not included in the existing two-dataset download. Place the
original Airfoil200 `meta.json`, `train.tfrecord`, `valid.tfrecord`, and
`test.tfrecord` in `data/airfoil/raw/`, then run:

```bash
python scripts/preprocess/preprocess_airfoil.py
python scripts/run_experiments/run_airfoil_benchmarks.py
```

For exact checkpoint evaluation, reuse the original normalization and spectral
artifacts (`normalization.json`, `airflow200_static.npz`,
`airflow200_modes169.npz`) in `data/airfoil/artifacts/`. Recomputing eigenspaces
can change eigenvector signs/bases; regenerated bases are suitable for fresh
training but are not guaranteed compatible with old checkpoints.

Each training command accepts `--help`, model selection, output paths, and
`--force`. Training saves validation-selected checkpoints and rollout metrics.
The shipped curves preserve the original completed runs; rerunning training
does not overwrite the bundled paper arrays.

## Source and verification

The final PFLOTRAN and Airfoil implementations live in
`src/tcwgno/pflotran_benchmark/` and `src/tcwgno/airfoil_benchmark/`.
SI uses `src/tcwgno/si_benchmark/`, with the selected K=5 SwiGLU factory.
The earlier 500k PFLOTRAN / K=6 SI source and historical tables remain available
for reference; `results/*/paper_summary/` and older unprefixed figures describe
that earlier release. Use `results/paper/` for current selections.

```bash
pytest
```

MIT license. See [LICENSE](LICENSE).
