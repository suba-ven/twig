# TWIG executable project

TWIG is the temporal wavelet graph operator reproducibility package for
PFLOTRAN, SI diffusion, and Airfoil.

From this directory, install with `pip install -e '.[test]'` and reproduce the
paper with `./run`. Datasets live in `../data/`; frozen reproduction inputs
are bundled in `results/paper/`. Outputs default to `results/codeocean/`
locally or `/results` in Code Ocean. Set `RESULTS_DIR` to override.

See the [repository README](../README.md) for the full workflow and
[data setup](../data/README.md) for dataset preparation.
