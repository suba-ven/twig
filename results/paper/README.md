# Frozen final paper curves

Each NPZ has nine display-name keys and arrays shaped `(3, rollout_steps)`:
PFLOTRAN 60, SI diffusion 100, Airfoil 180. Rows follow the seeds/runs listed
in `configs/paper_selections.json`. These are copied without recomputation from
`final-paper-figures/*_nine_model_curves.npz`.

`vis/plot_selected_results.py` plots them and exports summary statistics.
The default uses sample SD (ddof=1); SI notebook 45 and its table use population
SD (ddof=0). Model selection and means are identical under either convention.

The source paths in the manifest are provenance only. No original cluster
filesystem is required to plot these arrays. Raw simulation data and trained
weights are not bundled.
