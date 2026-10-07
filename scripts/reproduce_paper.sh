#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if [[ -z "${RESULTS_DIR:-}" ]]; then
    if [[ -d /results ]]; then
        RESULTS_DIR=/results
    else
        RESULTS_DIR="$REPO_ROOT/results/codeocean"
    fi
fi
export RESULTS_DIR
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/twig-matplotlib-${UID}}"
exec "${PYTHON:-python}" vis/reproduce_paper.py --output "$RESULTS_DIR" "$@"
