# TC-WGNO reproducibility package

Code and experiment records for **Temporal-Causal Wavelet Graph Neural Operator
(TC-WGNO)**. The package reproduces the capacity-matched PFLOTRAN H=10 benchmark
and the SI-diffusion benchmark reported in the accompanying paper.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

PyTorch Geometric must match the installed PyTorch/CUDA build; see its official
installation selector if the wheel selected by `pip` is unsuitable.

## Data

Download and extract both paper datasets from
[Hugging Face](https://huggingface.co/datasets/subaven/tc-wgno-benchmark-data):

```bash
./scripts/download_data.sh
```

The script uses resumable `wget` downloads and installs the files under
`data/pflotran/raw/` and `data/si_diffusion/`. To fetch only one dataset, pass
`pflotran` or `si_diffusion`. See [data/README.md](data/README.md) for archive
contents, manual commands, and validation.

## Reproduce experiments

All commands accept `--help`. A single run is written to
`results/<dataset>/<model>_seed<seed>/` with its configuration, checkpoint,
history, metrics, predictions/rollouts, and model metadata.

```bash
# Paper protocol: seeds 42, 43, 44; 30 epochs; capacity matched near 500k params
python scripts/run_experiments/run_pflotran_benchmarks.py data/pflotran/raw/*.h5
python scripts/run_experiments/run_si_benchmarks.py

# Evaluate or plot saved runs
python scripts/evaluate/evaluate_pflotran.py --help
python vis/plot_pflotran.py --help
```

The PFLOTRAN suite contains TC-WGNO (K=5 and K=7), Graph WNO, Graph FNO,
MeshGraphNet, GAT, GATv2, GPS Transformer, RNN, and RNN-GNN Fusion. Model names
and factories live in `tcwgno.utils.model_registry`.

The SI suite is the direct 14-to-14, three-seed benchmark at the
70,224-parameter reference capacity. It includes the reported TC-WGNO/SwiGLU,
standard TC-WGNO K=6 and K=2, Graph FNO, Graph WNO, GPS, GATv2, GAT,
MeshGraphNet, RNN-GNN Fusion, and RNN runs. It uses the original model classes,
optimizer, early stopping, scenario split, and block-rollout evaluation. The
downloadable first-100 package supports an 86-frame forecast after its 14-frame
context; the full legacy array supports the original requested 100-frame
forecast.

The exact exported paper tables are under `results/*/paper_summary/`, and the
corresponding publication figures are under `figures/`. These are copied from
the final experiment folders without recomputation.

## Verification

```bash
pytest
```

The source tree is a cleaned, importable copy derived from the authors' final
experiment code. Original notebooks, datasets, checkpoints, and experiment
directories are not required for package import and were not modified.

## License

MIT. See [LICENSE](LICENSE).
