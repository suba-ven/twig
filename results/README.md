# Current and archival results

`paper/` is the authoritative frozen reference for the current manuscript and
the default Code Ocean reproduction workflow. Generated outputs are written to
`/results` on Code Ocean or `results/codeocean/` locally.

`pflotran/paper_summary/` and `si_diffusion/paper_summary/` are **archival**
exports from the earlier release (including 500k PFLOTRAN and K=6 SI results).
They are retained for provenance and compatibility with historical analyses;
the default reproduction workflow does not read them. Do not substitute these
tables for the current nine-model selections in `configs/paper_selections.json`.
No historical results have been deleted.

## Generated experiment artifact contract

Each `<dataset>/<experiment>/` directory is self-contained and records:

- `config.json`: all data, model, optimizer, seed, and environment settings;
- `metrics.json`: scalar validation, test, and rollout metrics;
- `history.csv`: per-epoch train/validation losses and learning rate;
- `checkpoint.pt`: best model state and checkpoint metadata;
- `predictions.npz` and `rollouts.npz`: compressed predictions and targets;
- `model_info.json`: model name, exact trainable parameter count, and versions.

Generated binary artifacts are ignored by default. Release them through a DOI-backed
archive and place its checksum/URL in this file before publication.

`si_diffusion/paper_summary/` additionally contains the exact per-seed CSV/JSON
records copied from the three source run families used in the earlier release: base graph
and recurrent models, operator baselines, and the SwiGLU TWIG run. The
consolidated table is `si_final_clean_rollout_summary.csv`.
