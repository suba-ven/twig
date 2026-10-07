#!/usr/bin/env python
"""Prepare normalization and the 128/169-mode bases from original Airfoil200."""
import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from twig.airfoil_benchmark import config, prepare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('../data/airfoil/raw'))
    p.add_argument('--artifacts', type=Path, default=Path('../data/airfoil/artifacts'))
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    prepare.CONFIG = replace(config.CONFIG, data_dir=a.data, artifact_dir=a.artifacts)
    source = prepare.prepare(force=a.force)
    target = a.artifacts/'airflow200_modes169.npz'
    if target.exists() and not a.force: return
    with np.load(source) as data:
        pos = np.asarray(data['pos'], dtype=np.float64)
        src, dst = data['edge_index']
        weights = 1 / np.maximum(np.linalg.norm(pos[dst]-pos[src], axis=1), 1e-8)
        weights /= max(weights.mean(), 1e-12)
        adjacency = sparse.coo_matrix((weights, (src, dst)), shape=(len(pos),len(pos))).tocsr()
    adjacency = 0.5*(adjacency+adjacency.T)
    degree = np.asarray(adjacency.sum(1)).ravel()
    inv = sparse.diags(1/np.sqrt(np.maximum(degree,1e-12)))
    laplacian = sparse.eye(len(pos),format='csr')-inv@adjacency@inv
    evals, modes = eigsh(laplacian,k=169,which='SM',tol=1e-7)
    order = np.argsort(evals)
    np.savez_compressed(target, modes=modes[:,order].astype(np.float32),evals=evals[order].astype(np.float32))


if __name__ == '__main__':
    main()
