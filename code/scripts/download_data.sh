#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_BASE_URL="https://huggingface.co/datasets/subaven/twig-benchmark-data/resolve/main"
DOWNLOAD_DIR="${REPO_ROOT}/../data/downloads"
PFLOTRAN_DIR="${REPO_ROOT}/../data/pflotran/raw"
SI_DIR="${REPO_ROOT}/../data/si_diffusion"

usage() {
    echo "Usage: $0 [all|pflotran|si_diffusion]"
}

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Required command not found: $1" >&2
        exit 1
    fi
}

download_zip() {
    local filename="$1"
    mkdir -p "${DOWNLOAD_DIR}"
    echo "Downloading ${filename} ..."
    wget --continue --content-disposition \
        --output-document="${DOWNLOAD_DIR}/${filename}" \
        "${DATASET_BASE_URL}/${filename}?download=true"
}

download_pflotran() {
    local filename="pflotran_paper_data.zip"
    local staging
    download_zip "${filename}"
    staging="$(mktemp -d)"
    trap 'rm -rf "${staging}"' RETURN
    unzip -q -o "${DOWNLOAD_DIR}/${filename}" -d "${staging}"
    mkdir -p "${PFLOTRAN_DIR}"

    local count=0
    while IFS= read -r -d '' source; do
        cp -f "${source}" "${PFLOTRAN_DIR}/$(basename "${source}")"
        count=$((count + 1))
    done < <(find "${staging}" -type f \( -iname '*.h5' -o -iname '*.hdf5' \) -print0)

    if (( count == 0 )); then
        echo "No HDF5 files found in ${filename}." >&2
        exit 1
    fi
    echo "Installed ${count} PFLOTRAN files in ${PFLOTRAN_DIR}"
    rm -rf "${staging}"
    trap - RETURN
}

download_si_diffusion() {
    local filename="si_diffusion_paper_data.zip"
    local staging
    download_zip "${filename}"
    staging="$(mktemp -d)"
    trap 'rm -rf "${staging}"' RETURN
    unzip -q -o "${DOWNLOAD_DIR}/${filename}" -d "${staging}"
    mkdir -p "${SI_DIR}"

    local required=(
        "si_diffusion_data.pt"
        "si_diffusion_graph_edges.pt"
        "si_diffusion_node_coordinates.csv"
    )
    local name source
    for name in "${required[@]}"; do
        source="$(find "${staging}" -type f -name "${name}" -print -quit)"
        if [[ -z "${source}" ]]; then
            echo "Missing ${name} in ${filename}." >&2
            exit 1
        fi
        cp -f "${source}" "${SI_DIR}/${name}"
    done

    echo "Installed SI diffusion files in ${SI_DIR}"
    rm -rf "${staging}"
    trap - RETURN
}

require_command wget
require_command unzip

case "${1:-all}" in
    all)
        download_pflotran
        download_si_diffusion
        ;;
    pflotran)
        download_pflotran
        ;;
    si_diffusion|si)
        download_si_diffusion
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
