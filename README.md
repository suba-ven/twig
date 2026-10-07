# TWIG: three-dataset reproducibility package

Code and selected experiment results for **TWIG**, the temporal wavelet graph
operator, covering **PFLOTRAN,
SI diffusion, and Airfoil**. The default training commands now follow the final
nine-model comparisons.

## Install

```bash
cd code
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

The package is imported as `twig` (for example, `from twig.models import SATWIG3D`).
Reinstall the editable package after updating an existing checkout.

Run commands from the `code/` directory. PyTorch Geometric must match your
PyTorch/CUDA installation. Airfoil uses TFRecord files and the `tfrecord` package.

## Reproduce the paper

From `code/`, with Python 3.10 or newer:

```bash
pip install -e .
bash scripts/download_data.sh
bash scripts/reproduce_paper.sh
```

The data download is optional for this default workflow: it regenerates the
manuscript figures and tables from the frozen results in `results/paper/`,
using `configs/paper_selections.json`. It performs no training or checkpoint
inference and needs no GPU or network access after dependencies are installed.
PFLOTRAN and SI diffusion downloads are hosted on
[Hugging Face](https://huggingface.co/datasets/subaven/twig-benchmark-data).
See [data/README.md](data/README.md) for downloads and preprocessing when
running new experiments. Bash, `wget`, and `unzip` are needed for downloading.

Outputs go to `./results/codeocean/` locally, or an explicit directory:

```bash
RESULTS_DIR=/path/to/output bash scripts/reproduce_paper.sh
```

The output includes `tables/manuscript_summary.csv` and `.tex` (27 selections),
the noise summary, extended-data plots, input checksums, package versions, and
all five manuscript figures under `figures/`:

- `pflotran-noise-robustness-twig.png`
- `si_diffusion_top4_errors_blue_twig.png`
- `figure_2_rollouts_three_datasets_pflotran_zoom_twig.pdf`
- `pflotran_top4_final_pressure_rmse_pa_uncapped_gray_edges_twig.png`
- `airfoil_top4_final_compact_centered_zoom15_no_rmse_twig.png`

Figures are rendered anew from compact numeric inputs. The exact original
exports in `figures/paper/` are for comparison and are never read by the run.
PDF/PNG rendering can vary slightly with Matplotlib and font versions.
Table SD uses ddof=0 for SI and ddof=1 for PFLOTRAN/Airfoil; rollout shading
uses ddof=1. The SI spatial figure shows selected-checkpoint absolute error;
the PFLOTRAN/Airfoil spatial figures show mean per-run test RMSE.

## Code Ocean

The repository follows the capsule layout: executable files are under `code/`,
datasets are under `data/`, and the top-level README and LICENSE document the
repository. Code Ocean manages `metadata/` and `environment/`.

Use `code/run` as the executable entry point (`/code/run` inside the capsule).
Pressing **Run** invokes this script. The frozen inputs remain alongside the
executable project in `code/results/paper/` (`/code/results/paper/` in the capsule).

In the capsule environment build, select Python 3.10+ and install the packages
in `code/requirements.txt` (plus setuptools>=68 and wheel for editable installation).
The normal local setup `pip install -e .` also installs runtime dependencies.
The entry-point script uses the checked-out `src/` package via `PYTHONPATH`, so it
requires no package installation or internet access at Run time.
No CUDA runtime is needed for the default frozen-results workflow.

Generated outputs use `${RESULTS_DIR:-/results}` when the capsule's `/results`
mount exists. Outside a capsule, absence of that mount selects
`./results/codeocean/`. An explicit `RESULTS_DIR` always takes precedence;
permission errors on an existing mount fail rather than silently redirecting.
`PYTHON=/path/to/python ./run` selects an interpreter if needed.

A lightweight integration check (no pytest installation required) is:

```bash
python tests/test_reproduction.py
```

## Selected experiments

[code/configs/paper_selections.json](code/configs/paper_selections.json) records all 27
model/dataset selections, parameter counts, hashed provenance identifiers,
selected field runs, and Airfoil's per-run training settings.

| Dataset | Context / direct forecast | Rollout | TWIG |
|---|---|---|---|
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
Airfoil uses 100 epochs with the saved per-model batch size 4/8, warmup
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
The selection manifest points to bundled arrays. Source-path identifiers are
SHA256 hashes of original cluster path strings; they are provenance markers,
not download URLs or file-content checksums. Original checkpoints are not bundled.

## Data and training

The existing downloader supplies PFLOTRAN and the shortened SI data package:

```bash
bash scripts/download_data.sh
python scripts/run_experiments/run_pflotran_benchmarks.py ../data/pflotran/raw/*.h5
python scripts/run_experiments/run_si_benchmarks.py --data /path/to/SI_equation_dataset.npy
```

The final SI 100-step rollout needs at least 114 states (14 context + 100
forecast). The first-100 download supports only 86 forecast steps and cannot
reproduce the final 100-step comparison. Supply the full original SI array for
the reported horizon. Edges and coordinates default to the downloaded files.

Airfoil data is not included in the existing two-dataset download. Place the
prepared Airfoil200 `meta.json`, `train.tfrecord`, `valid.tfrecord`, and
`test.tfrecord` in `../data/airfoil/raw/` using the download and 601-to-200-state
conversion commands in [data/README.md](data/README.md), then run:

```bash
python scripts/preprocess/preprocess_airfoil.py
python scripts/run_experiments/run_airfoil_benchmarks.py
```

For exact checkpoint evaluation, reuse the original normalization and spectral
artifacts (`normalization.json`, `airflow200_static.npz`,
`airflow200_modes169.npz`) in `../data/airfoil/artifacts/`. Recomputing eigenspaces
can change eigenvector signs/bases; regenerated bases are suitable for fresh
training but are not guaranteed compatible with old checkpoints.

Each training command accepts `--help`, model selection, output paths, and
`--force`. Training saves validation-selected checkpoints and rollout metrics.
The shipped curves preserve the original completed runs; rerunning training
does not overwrite the bundled paper arrays.

## Source and verification

The final PFLOTRAN and Airfoil implementations live in
`src/twig/pflotran_benchmark/` and `src/twig/airfoil_benchmark/`.
SI uses `src/twig/si_benchmark/`, with the selected K=5 SwiGLU factory.
The earlier 500k PFLOTRAN / K=6 SI source and historical tables remain available
for reference; `results/*/paper_summary/` and older unprefixed figures describe
that earlier release. Use `results/paper/` for current selections.

```bash
pytest
```

MIT license. See [LICENSE](LICENSE).
