# Data setup

The paper data are hosted at
[`subaven/tc-wgno-benchmark-data`](https://huggingface.co/datasets/subaven/tc-wgno-benchmark-data)
and remain excluded from Git. From the repository root, download both archives
with:

```bash
./scripts/download_data.sh
```

The downloads are resumable. Fetch a single archive with
`./scripts/download_data.sh pflotran` or
`./scripts/download_data.sh si_diffusion`.

## PFLOTRAN

`pflotran_paper_data.zip` is downloaded and its HDF5 scenario files are placed
under `data/pflotran/raw/`. Validate and create the split manifest with:

```bash
python scripts/preprocess/preprocess_pflotran.py data/pflotran/raw/*.h5
```

The preprocessor uses the same
scenario-level split and train-only normalization implemented in
`tcwgno.data.pflotran`; pass `--help` to select variables, history, horizon, and
split seed. It creates `dataset_info.json`, which is the input to training.

## SI diffusion

`si_diffusion_paper_data.zip` contains the first 100 timesteps of each of the 25
scenarios plus the graph metadata. The downloader installs:

```text
data/si_diffusion/si_diffusion_data.pt
data/si_diffusion/si_diffusion_graph_edges.pt
data/si_diffusion/si_diffusion_node_coordinates.csv
```

The packaged tensor has shape `(25, 100, 400, 2)`. Validate it and create its
manifest using the installed defaults:

```bash
python scripts/preprocess/preprocess_si_diffusion.py
```

The protocol uses infected-state channel 1 without rescaling, a fixed scenario
split of 19/3/3, history 14, and direct horizon 14. Because the downloadable
package contains 100 total frames per scenario, its longest valid forecast is
86 frames after the 14-frame context. The loader automatically uses that valid
length and records it in the generated manifest. Supplying the original
364-frame array retains the requested 100-frame block rollout.

The loader also remains backward-compatible with the original full
`SI_equation_dataset.npy` shape `(25*364, 400, 2)` when an explicit `--data`
path is supplied. Neither preprocessor mutates its input.

## Manual download

If the helper script cannot be used, the equivalent archive downloads are:

```bash
wget -c https://huggingface.co/datasets/subaven/tc-wgno-benchmark-data/resolve/main/pflotran_paper_data.zip
wget -c https://huggingface.co/datasets/subaven/tc-wgno-benchmark-data/resolve/main/si_diffusion_paper_data.zip
```

## Final three-dataset comparison

The downloader above still contains two datasets. Airfoil must be supplied
separately: use the original contiguous Airfoil200 TFRecords and `meta.json`
under `data/airfoil/raw/`, then run `python scripts/preprocess/preprocess_airfoil.py`.
The selected experiment uses H=20/F=20, all four channels, and 180 rollout steps.
Do not substitute the later stride-2 or stride-3 data.

The final SI plots forecast 100 states **after** 14 context states. The shortened
first-100 package is insufficient; pass the full `SI_equation_dataset.npy` to
the SI training command for a 100-step comparison. Frozen final result curves
for all three datasets can be plotted without any dataset download.
