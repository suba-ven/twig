# Frozen final paper curves

Each NPZ has nine display-name keys and arrays shaped `(3, rollout_steps)`:
PFLOTRAN 60, SI diffusion 100, Airfoil 180. Rows follow the seeds/runs listed
in `configs/paper_selections.json`. These are copied without recomputation from
`final-paper-figures/*_nine_model_curves.npz`.

`vis/plot_selected_results.py` plots them and exports summary statistics.
The default uses sample SD (ddof=1); SI notebook 45 and its table use population
SD (ddof=0). Model selection and means are identical under either convention.

The manifest references bundled arrays and retains hashed source-path IDs
for provenance. These IDs hash path strings, not checkpoint contents. No original
cluster filesystem is required to plot these arrays. Raw simulation data and trained
weights are not bundled.

## Frozen manuscript plotting inputs

The default `bash scripts/reproduce_paper.sh` also consumes:

- `pflotran_noise_summary.csv`: original clean-trained/noisy-test summary.
- `si_paper_fields.npz`: final-step absolute errors, coordinates, and graph edges
  for the four selected individual checkpoint predictions (100-step rollout).
- `airfoil_paper_fields.npz`: four-model final-step mean per-run test velocity
  RMSE and mesh geometry; no full trajectories or checkpoint weights.
- `pflotran_paper_fields.npz`: four-model final-step mean per-run test pressure
  RMSE in Pa, projected original top-face triangles, and coarse-node mapping.
  The projection is frozen without smoothing, clipping, or altering RMSE.
- `paper_plot_metadata.json`: original model ordering, scales, layout, and
  aggregation conventions for the spatial figures.
- `plot_inputs_sha256.json`: checksums validated before rendering.

These small inputs were extracted from the original `final-paper-figures`
exports. SI uses `si_diffusion_physical_data.npz` and the blue TWIG manifest;
Airfoil/PFLOTRAN use `three seed test rmse plots/spatial fields/` aggregates
and `final-final-plots/final_four_model_manifest_twig.json`. PFLOTRAN geometry
comes from `Domain/Vertices`, `Domain/Cells`, and `Domain/CoarseGraph` in the
original compressed HDF5. Noise data come from
`clean_trained_noisy_test_step60/noisy_test_summary.csv`.

Maintainers with those original exports can refresh inputs explicitly:

```bash
python scripts/preprocess/freeze_paper_plot_inputs.py \
  --source /path/to/final-paper-figures \
  --pflotran-geometry /path/to/compressed/M155_0m0_compressed.h5
```

This is an authoring step, not required by capsule users. Review any input and
checksum changes together. The original rollout NPZ checksums remain recorded
in `configs/paper_selections.json`. The default workflow never modifies this
folder or reads images from `figures/`.
