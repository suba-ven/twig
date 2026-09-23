from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .config import CONFIG
from .data import compute_training_stats, extract_static_graph


def prepare(force: bool = False) -> Path:
    out = CONFIG.artifact_dir
    out.mkdir(parents=True, exist_ok=True)
    target = out / "airflow200_static.npz"
    stats_path = out / "normalization.json"
    if target.exists() and stats_path.exists() and not force:
        return target

    stats = compute_training_stats(CONFIG.data_dir)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    pos2, edge_index, node_type = extract_static_graph(CONFIG.data_dir)
    pos3 = np.pad(pos2, ((0, 0), (0, 1))).astype(np.float32)
    pos_mean = pos3.mean(0, keepdims=True)
    pos_std = pos3.std(0, keepdims=True)
    pos_std[pos_std < 1.0e-8] = 1.0
    pos_norm = (pos3 - pos_mean) / pos_std

    src, dst = edge_index
    dist = np.linalg.norm(pos_norm[dst] - pos_norm[src], axis=1)
    weights = 1.0 / np.maximum(dist, 1.0e-8)
    adjacency = sp.coo_matrix((weights, (src, dst)), shape=(len(pos3), len(pos3))).tocsr()
    adjacency = 0.5 * (adjacency + adjacency.T)
    degree = np.asarray(adjacency.sum(1)).ravel()
    inv = 1.0 / np.sqrt(np.maximum(degree, 1.0e-12))
    laplacian = sp.eye(len(pos3)) - sp.diags(inv) @ adjacency @ sp.diags(inv)
    evals, modes = spla.eigsh(laplacian, k=CONFIG.n_modes, which="SM", tol=1.0e-6)
    order = np.argsort(evals)
    np.savez_compressed(target, pos=pos_norm, edge_index=edge_index, node_type=node_type,
                        evals=evals[order].astype(np.float32), modes=modes[:, order].astype(np.float32))
    return target


if __name__ == "__main__":
    print(prepare())
