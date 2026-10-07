# Data setup

PFLOTRAN and SI diffusion data are hosted at
[`subaven/twig-benchmark-data`](https://huggingface.co/datasets/subaven/twig-benchmark-data)
and remain excluded from Git. From the `code/` directory, download both archives
with:

```bash
bash scripts/download_data.sh
```

The downloads are resumable. Fetch a single archive with
`bash scripts/download_data.sh pflotran` or
`bash scripts/download_data.sh si_diffusion`.

## PFLOTRAN

`pflotran_paper_data.zip` is downloaded and its HDF5 scenario files are placed
under `../data/pflotran/raw/`. Validate and create the split manifest with:

```bash
python scripts/preprocess/preprocess_pflotran.py ../data/pflotran/raw/*.h5
```

The preprocessor uses the same
scenario-level split and train-only normalization implemented in
`twig.data.pflotran`; pass `--help` to select variables, history, horizon, and
split seed. It creates `dataset_info.json`, which is the input to training.

## SI diffusion

`si_diffusion_paper_data.zip` contains the first 100 timesteps of each of the 25
scenarios plus the graph metadata. The downloader installs:

```text
../data/si_diffusion/si_diffusion_data.pt
../data/si_diffusion/si_diffusion_graph_edges.pt
../data/si_diffusion/si_diffusion_node_coordinates.csv
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
wget -c https://huggingface.co/datasets/subaven/twig-benchmark-../data/resolve/main/pflotran_paper_data.zip
wget -c https://huggingface.co/datasets/subaven/twig-benchmark-../data/resolve/main/si_diffusion_paper_data.zip
```

## Final three-dataset comparison

Airfoil comes from the public [MeshGraphNets dataset](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets),
not the project's Hugging Face dataset. Download the original Airfoil files
using the public bucket from its
[official downloader](https://github.com/google-deepmind/deepmind-research/blob/master/meshgraphnets/download_dataset.sh):

```bash
mkdir -p ../data/airfoil/official
for file in meta.json train.tfrecord valid.tfrecord test.tfrecord; do
  wget -c -O "../data/airfoil/official/$file" \
    "https://storage.googleapis.com/dm-meshgraphnets/airfoil/$file"
done
python scripts/preprocess/prepare_airfoil200.py \
  --source ../data/airfoil/official --output ../data/airfoil/raw
python scripts/preprocess/preprocess_airfoil.py \
  --data ../data/airfoil/raw --artifacts ../data/airfoil/artifacts
```

The public release contains 601 states per trajectory. `prepare_airfoil200.py`
extracts states 0:200 with stride 1, preserving split membership, record order,
static geometry, and the original time interval. It streams records and refuses
to overwrite an existing output directory. The first record of each prepared
original split was checked against this prefix during repository validation.

The selected experiment requires these contiguous Airfoil200 records:
H=20/F=20, all four channels, and 180 rollout steps. The loader validates
200 states and rejects declared temporal subsampling. Do not substitute
stride-2 or stride-3 data. For exact evaluation of old checkpoints, retain
the original normalization and spectral artifacts; recomputed eigenvectors
may have different signs or bases. Preprocessing is for fresh training.

The final SI plots forecast 100 states **after** 14 context states. The shortened
first-100 package is insufficient; pass the full `SI_equation_dataset.npy` to
the SI training command for a 100-step comparison. Frozen final result curves
for all three datasets can be plotted without any dataset download.

## Storage

Never commit large datasets, downloaded archives, preprocessing artifacts, or
checkpoints to Git. Keep them in ignored `../data/` directories or attach them as
Code Ocean data assets when running new experiments. The default frozen-results
workflow needs none of these downloads. On Sherlock, use project scratch for
large ../data/preprocessing and compute nodes for preprocessing/training; pass
explicit input/output paths as appropriate.
