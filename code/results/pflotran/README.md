# Archival results

These historical `paper_summary/` tables are retained for provenance, not used
by the current manuscript reproduction workflow. Use `../paper/` and
`configs/paper_selections.json` for current selections. See `../README.md`.

`paper_summary/archival_curves.npz` freezes the original three-run curves
used by the retained historical figures; its adjacent JSON records the checksum.
Regenerate these figures with current TWIG labels using:

```bash
python vis/plot_archival_results.py --output figures
```

This archival workflow is separate from the manuscript's default reproduction.
Historical source paths containing superseded model identifiers are retained
as `archived-source-sha256:` IDs (hashes of path strings, not file contents).
