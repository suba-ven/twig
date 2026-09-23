from __future__ import annotations

import json
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# -----------------------------------------------------------------------------
# File/path helpers
# -----------------------------------------------------------------------------

def _extract_zip_if_needed(input_path: str, extract_dir: Optional[str] = None) -> List[str]:
    p = Path(input_path).expanduser()
    if p.suffix.lower() == ".zip":
        if extract_dir is None:
            extract_dir = str(p.with_suffix(""))
        extract_dir = Path(extract_dir).expanduser()
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(p, "r") as zf:
            zf.extractall(extract_dir)
        h5s = sorted(str(x) for x in extract_dir.rglob("*.h5"))
        if not h5s:
            raise ValueError(f"No .h5 files found after extracting {input_path}")
        return h5s

    if p.is_dir():
        h5s = sorted(str(x) for x in p.rglob("*.h5"))
        if not h5s:
            raise ValueError(f"No .h5 files found under directory {input_path}")
        return h5s

    if p.is_file() and p.suffix.lower() == ".h5":
        return [str(p)]

    raise ValueError(f"Could not resolve .h5 paths from {input_path}")


def resolve_h5_paths(h5_paths: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(h5_paths, str):
        return _extract_zip_if_needed(h5_paths)
    out = []
    for p in h5_paths:
        out.extend(_extract_zip_if_needed(str(p)))
    return sorted(out)


def _time_sort_key(k: str) -> int:
    s = k.strip()
    try:
        return int(s.split()[0])
    except Exception:
        return 10**18


def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def resolve_var_name(available: Sequence[str], requested: str) -> str:
    if requested in available:
        return requested

    avail = list(available)
    reqn = _norm_key(requested)
    norm_map: Dict[str, List[str]] = {}
    for k in avail:
        norm_map.setdefault(_norm_key(k), []).append(k)

    if reqn in norm_map and len(norm_map[reqn]) == 1:
        return norm_map[reqn][0]

    if "liquid pressure" in requested.lower():
        toks = ["liquid", "pressure"]
    elif "total sal" in requested.lower():
        toks = ["total", "sal"]
    else:
        toks = [t for t in re.split(r"[^A-Za-z0-9]+", requested.lower()) if t]

    candidates = []
    for k in avail:
        kl = k.lower()
        if all(t in kl for t in toks):
            candidates.append(k)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous variable name {requested!r}. Candidates: {candidates}")
    raise ValueError(f"Variable {requested!r} not found. Available keys: {avail[:20]}")


def list_time_keys(f: h5py.File) -> List[str]:
    exclude = {"Domain", "CoarseningInfo", "model_params", "CoarseGraph"}
    keys = [k for k in f.keys() if k not in exclude]
    return sorted(keys, key=_time_sort_key)


def read_graph_from_file_3d(h5_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        dom = f["Domain"]
        if "CoarseGraph" not in dom:
            raise ValueError(f"{h5_path}: expected Domain/CoarseGraph")
        cg = dom["CoarseGraph"]
        if "node_pos" not in cg or "edge_index" not in cg:
            raise ValueError(f"{h5_path}: missing node_pos or edge_index")
        node_pos = cg["node_pos"][:].astype(np.float32, copy=False)
        edge_index = cg["edge_index"][:].astype(np.int64, copy=False)

    if node_pos.ndim != 2 or node_pos.shape[1] != 3:
        raise ValueError(f"{h5_path}: expected node_pos shape (N,3), got {node_pos.shape}")
    if edge_index.ndim != 2 or 2 not in edge_index.shape:
        raise ValueError(f"{h5_path}: expected edge_index shape (E,2) or (2,E), got {edge_index.shape}")
    if edge_index.shape[0] == 2:
        edge_index = edge_index.T
    return node_pos, edge_index


def read_model_params(h5_path: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        if "model_params" in f:
            arr = f["model_params"][:]
        elif "/model_params" in f:
            arr = f["/model_params"][:]
        else:
            found = None
            for k in f.keys():
                obj = f[k]
                if isinstance(obj, h5py.Group) and "model_params" in obj:
                    found = obj["model_params"][:]
                    break
            if found is None:
                raise ValueError(f"{h5_path}: could not find model_params")
            arr = found
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"{h5_path}: model_params must be 1D, got {arr.shape}")
    return arr


# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------

@dataclass
class QuantileStats:
    q_low_pct: float
    q_high_pct: float
    q_low: float
    q_high: float
    eps: float = 1e-12

    def normalize(self, x: np.ndarray) -> np.ndarray:
        lo, hi = float(self.q_low), float(self.q_high)
        denom = max(hi - lo, self.eps)
        y = (x - lo) / denom
        y = np.clip(y, 0.0, 1.0)
        return (2.0 * y - 1.0).astype(np.float32, copy=False)

    def denormalize(self, y: np.ndarray) -> np.ndarray:
        lo, hi = float(self.q_low), float(self.q_high)
        x01 = (y + 1.0) * 0.5
        return (x01 * (hi - lo) + lo).astype(np.float32, copy=False)

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "QuantileStats":
        return cls(
            q_low_pct=float(d["q_low_pct"]),
            q_high_pct=float(d["q_high_pct"]),
            q_low=float(d["q_low"]),
            q_high=float(d["q_high"]),
            eps=float(d.get("eps", 1e-12)),
        )


def compute_quantile_stats(values: np.ndarray, q_low: float, q_high: float) -> QuantileStats:
    if values.size == 0:
        raise ValueError("empty values array")
    ql = float(np.nanpercentile(values, q_low))
    qh = float(np.nanpercentile(values, q_high))
    if not np.isfinite(ql) or not np.isfinite(qh):
        raise ValueError("non-finite quantiles")
    if abs(qh - ql) < 1e-15:
        qh = ql + 1e-6
    return QuantileStats(q_low_pct=float(q_low), q_high_pct=float(q_high), q_low=ql, q_high=qh)


def _compute_stats_across_files(
    h5_paths: Sequence[str],
    var_names: Sequence[str],
    q_low: float,
    q_high: float,
) -> Tuple[Dict[str, QuantileStats], List[QuantileStats]]:
    """Compute normalization statistics from training scenario files only."""
    values_per_var = {vn: [] for vn in var_names}
    params_all = []

    for p in h5_paths:
        with h5py.File(p, "r") as f:
            tks = list_time_keys(f)
            if not tks:
                raise ValueError(f"{p}: no time keys found")
            avail = list(f[tks[0]].keys())
            resolved = {vn: resolve_var_name(avail, vn) for vn in var_names}
            for tk in tks:
                g = f[tk]
                for vn in var_names:
                    arr = g[resolved[vn]][:].astype(np.float32, copy=False)
                    values_per_var[vn].append(arr.reshape(-1))
        params_all.append(read_model_params(p).reshape(-1))

    stats_vars = {}
    for vn in var_names:
        stats_vars[vn] = compute_quantile_stats(
            np.concatenate(values_per_var[vn], axis=0), q_low, q_high
        )

    params_all = np.stack(params_all, axis=0)
    stats_params = [compute_quantile_stats(params_all[:, j], q_low, q_high) for j in range(params_all.shape[1])]
    return stats_vars, stats_params


@dataclass
class PositionStats:
    center: List[float]
    scale: float
    method: str = "center_unit_box"
    eps: float = 1e-12

    def normalize(self, pos: np.ndarray) -> np.ndarray:
        center = np.asarray(self.center, dtype=np.float32)
        return ((pos.astype(np.float32) - center) / max(float(self.scale), self.eps)).astype(np.float32)

    @classmethod
    def from_node_pos(cls, node_pos: np.ndarray) -> "PositionStats":
        lo = node_pos.min(axis=0)
        hi = node_pos.max(axis=0)
        center = 0.5 * (lo + hi)
        scale = float(np.max(hi - lo))
        if scale <= 0:
            scale = 1.0
        return cls(center=center.astype(float).tolist(), scale=scale)

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "PositionStats":
        return cls(
            center=[float(x) for x in d["center"]],
            scale=float(d["scale"]),
            method=str(d.get("method", "center_unit_box")),
            eps=float(d.get("eps", 1e-12)),
        )


def _stats_to_jsonable(stats_vars: Dict[str, QuantileStats], stats_params: List[QuantileStats]) -> Dict[str, object]:
    return {
        "vars": {vn: asdict(st) for vn, st in stats_vars.items()},
        "params": [asdict(st) for st in stats_params],
    }


def stats_from_jsonable(d: Dict[str, object]) -> Tuple[Dict[str, QuantileStats], List[QuantileStats]]:
    stats_vars = {vn: QuantileStats.from_dict(v) for vn, v in d["vars"].items()}
    stats_params = [QuantileStats.from_dict(v) for v in d["params"]]
    return stats_vars, stats_params


# -----------------------------------------------------------------------------
# Scenario split
# -----------------------------------------------------------------------------

def _split_scenarios(n: int, train_frac: float, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_train = min(max(n_train, 0), n)
    n_val = min(max(n_val, 0), n - n_train)
    return (
        idx[:n_train].tolist(),
        idx[n_train:n_train + n_val].tolist(),
        idx[n_train + n_val:].tolist(),
    )


def _atomic_write_json(path: Union[str, Path], payload: Dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=str(path.parent), delete=False) as f:
        json.dump(payload, f, indent=2)
        tmp = f.name
    os.replace(tmp, path)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class PFLOTRANBlockDataset3D(Dataset):
    """PFLOTRAN block forecasting dataset.

    Each item returns
      x:      (H, N, C)
      y:      (F, N, C)
      params: (P,)
      meta:   dictionary with scenario and time information

    Variable stats should be computed on training scenarios only. Node positions
    are normalized once via PositionStats and the normalized coordinates are what
    downstream models see. Raw coordinates are preserved in `raw_node_pos`.
    """

    def __init__(
        self,
        h5_paths: Sequence[str],
        var_names: Sequence[str],
        history: int,
        forecast_horizon: int,
        stats_vars: Optional[Dict[str, QuantileStats]],
        stats_params: Optional[List[QuantileStats]],
        pos_stats: Optional[PositionStats] = None,
        normalize: bool = True,
    ):
        if history < 1:
            raise ValueError("history must be >= 1")
        if forecast_horizon < 1:
            raise ValueError("forecast_horizon must be >= 1")

        self.h5_paths = list(h5_paths)
        self.var_names = tuple(var_names)
        self.history = int(history)
        self.forecast_horizon = int(forecast_horizon)
        self.normalize = bool(normalize)
        self.stats_vars = stats_vars
        self.stats_params = stats_params

        pos0, e0 = read_graph_from_file_3d(self.h5_paths[0])
        self.raw_node_pos = pos0.astype(np.float32)
        self.pos_stats = pos_stats or PositionStats.from_node_pos(self.raw_node_pos)
        self.node_pos = self.pos_stats.normalize(self.raw_node_pos)
        self.edge_index = e0.astype(np.int64)
        self.N = int(self.node_pos.shape[0])

        for p in self.h5_paths[1:]:
            pos, e = read_graph_from_file_3d(p)
            if pos.shape != self.raw_node_pos.shape or not np.allclose(pos, self.raw_node_pos):
                raise ValueError(f"Graph node_pos mismatch between {self.h5_paths[0]} and {p}")
            if e.shape != self.edge_index.shape or not np.array_equal(e, self.edge_index):
                raise ValueError(f"Graph edge_index mismatch between {self.h5_paths[0]} and {p}")

        self.time_keys_per_s: List[List[str]] = []
        self.params_per_s: List[np.ndarray] = []
        self.var_key_map_per_s: List[Dict[str, str]] = []

        for p in self.h5_paths:
            with h5py.File(p, "r") as f:
                tks = list_time_keys(f)
                if len(tks) < self.history + self.forecast_horizon:
                    raise ValueError(
                        f"{p}: need at least H+F={self.history + self.forecast_horizon} timesteps, got {len(tks)}"
                    )
                avail = list(f[tks[0]].keys())
                key_map = {vn: resolve_var_name(avail, vn) for vn in self.var_names}
            self.time_keys_per_s.append(tks)
            self.params_per_s.append(read_model_params(p))
            self.var_key_map_per_s.append(key_map)

        self.index: List[Tuple[int, int]] = []
        for s, tks in enumerate(self.time_keys_per_s):
            # t is the final observed index. Targets are t+1, ..., t+F.
            for t in range(self.history - 1, len(tks) - self.forecast_horizon):
                self.index.append((s, t))

        self._open_files: Dict[int, h5py.File] = {}

    def __len__(self):
        return len(self.index)

    def _get_file(self, scenario_idx: int) -> h5py.File:
        if scenario_idx not in self._open_files:
            self._open_files[scenario_idx] = h5py.File(self.h5_paths[scenario_idx], "r")
        return self._open_files[scenario_idx]

    def _read_state(self, f: h5py.File, scenario_idx: int, time_key: str) -> np.ndarray:
        g = f[time_key]
        feats = []
        for vn in self.var_names:
            key = self.var_key_map_per_s[scenario_idx][vn]
            arr = g[key][:].astype(np.float32, copy=False)
            if arr.shape[0] != self.N:
                raise ValueError(f"{self.h5_paths[scenario_idx]}: {vn} has N={arr.shape[0]} vs {self.N}")
            if self.normalize and self.stats_vars is not None:
                arr = self.stats_vars[vn].normalize(arr)
            feats.append(arr)
        return np.stack(feats, axis=-1)  # (N,C)

    def __getitem__(self, idx: int):
        s, t = self.index[idx]
        f = self._get_file(s)
        tks = self.time_keys_per_s[s]

        x = np.stack(
            [self._read_state(f, s, tks[tt]) for tt in range(t - self.history + 1, t + 1)],
            axis=0,
        ).astype(np.float32)
        y = np.stack(
            [self._read_state(f, s, tks[tt]) for tt in range(t + 1, t + 1 + self.forecast_horizon)],
            axis=0,
        ).astype(np.float32)

        params = self.params_per_s[s].copy().astype(np.float32)
        if self.normalize and self.stats_params is not None:
            pnorm = []
            for i, st in enumerate(self.stats_params):
                pnorm.append(st.normalize(np.asarray(params[i], dtype=np.float32)))
            params = np.stack(pnorm, axis=0).astype(np.float32, copy=False)

        meta = {
            "scenario_idx": int(s),
            "h5_path": self.h5_paths[s],
            "t_idx": int(t),
            "time_key": tks[t],
            "target_start_key": tks[t + 1],
            "target_end_key": tks[t + self.forecast_horizon],
        }
        return x, y, params, meta

    def close(self):
        for f in self._open_files.values():
            try:
                f.close()
            except Exception:
                pass
        self._open_files = {}

    def __del__(self):
        self.close()


def _canonical_edge_index_np(edge_index: np.ndarray) -> np.ndarray:
    if edge_index.shape[0] == 2:
        return edge_index.astype(np.int64)
    if edge_index.shape[1] == 2:
        return edge_index.T.astype(np.int64)
    raise ValueError(f"Invalid edge_index shape {edge_index.shape}")


def _collate_block_3d(batch, node_pos: np.ndarray, edge_index: np.ndarray):
    xs, ys, ps, metas = zip(*batch)
    x = torch.from_numpy(np.stack(xs, axis=0)).float()  # (B,H,N,C)
    y = torch.from_numpy(np.stack(ys, axis=0)).float()  # (B,F,N,C)
    params = torch.from_numpy(np.stack(ps, axis=0)).float()
    node_pos_t = torch.from_numpy(node_pos).float()
    edge_index_t = torch.from_numpy(_canonical_edge_index_np(edge_index)).long()
    meta = {
        "scenario_idx": torch.tensor([m["scenario_idx"] for m in metas], dtype=torch.long),
        "h5_path": [m["h5_path"] for m in metas],
        "t_idx": torch.tensor([m["t_idx"] for m in metas], dtype=torch.long),
        "time_key": [m["time_key"] for m in metas],
        "target_start_key": [m["target_start_key"] for m in metas],
        "target_end_key": [m["target_end_key"] for m in metas],
    }
    return x, y, params, meta, node_pos_t, edge_index_t


def build_block_dataloaders_3d(
    h5_paths: Union[str, Sequence[str]],
    var_names: Sequence[str] = ("Liquid Pressure [Pa]", "Total Sal [M]"),
    history: int = 10,
    forecast_horizon: int = 10,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = False,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 0,
    q_low: float = 1.0,
    q_high: float = 99.0,
    normalize: bool = True,
    split_stats_path: Optional[Union[str, Path]] = None,
):
    """Build scenario-split PFLOTRAN block dataloaders.

    This fixes two common leakage issues in one-step loaders:
      1. train/val/test splits are by scenario file, not by overlapping windows;
      2. variable/parameter normalization statistics are computed on training scenarios only.

    Position normalization is deterministic and graph-wide: raw coordinates are centered
    by the bounding-box midpoint and divided by the largest bounding-box side length.
    The normalized positions are returned in every batch and used by downstream models.
    """
    all_paths = resolve_h5_paths(h5_paths)
    if len(all_paths) < 3:
        raise ValueError(f"Expected at least 3 h5 scenario files, got {len(all_paths)}")

    cache = None
    if split_stats_path is not None:
        split_stats_path = Path(split_stats_path)
        if split_stats_path.is_file():
            with open(split_stats_path, "r") as f:
                cache = json.load(f)

    if cache is not None:
        if cache.get("history") != int(history) or cache.get("forecast_horizon") != int(forecast_horizon):
            raise ValueError("Cached split/stats were created with different H/F.")
        if list(cache.get("var_names", [])) != list(var_names):
            raise ValueError("Cached split/stats were created with different var_names.")
        train_idx = [int(i) for i in cache["splits"]["train_idx"]]
        val_idx = [int(i) for i in cache["splits"]["val_idx"]]
        test_idx = [int(i) for i in cache["splits"]["test_idx"]]
        stats_vars, stats_params = stats_from_jsonable(cache["stats"])
        pos_stats = PositionStats.from_dict(cache["position_stats"])
    else:
        train_idx, val_idx, test_idx = _split_scenarios(len(all_paths), train_frac, val_frac, seed)
        train_paths = [all_paths[i] for i in train_idx]
        stats_vars, stats_params = _compute_stats_across_files(train_paths, var_names, q_low, q_high)
        raw_pos, _ = read_graph_from_file_3d(all_paths[0])
        pos_stats = PositionStats.from_node_pos(raw_pos)

    train_paths = [all_paths[i] for i in train_idx]
    val_paths = [all_paths[i] for i in val_idx]
    test_paths = [all_paths[i] for i in test_idx]

    ds_train = PFLOTRANBlockDataset3D(
        train_paths, var_names, history, forecast_horizon, stats_vars, stats_params, pos_stats, normalize
    )
    ds_val = PFLOTRANBlockDataset3D(
        val_paths, var_names, history, forecast_horizon, stats_vars, stats_params, pos_stats, normalize
    )
    ds_test = PFLOTRANBlockDataset3D(
        test_paths, var_names, history, forecast_horizon, stats_vars, stats_params, pos_stats, normalize
    )

    collate_fn = lambda batch: _collate_block_3d(batch, ds_train.node_pos, ds_train.edge_index)

    train_loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, collate_fn=collate_fn, drop_last=False
    )
    val_loader = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, collate_fn=collate_fn, drop_last=False
    )
    test_loader = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, collate_fn=collate_fn, drop_last=False
    )

    info = {
        "history": int(history),
        "forecast_horizon": int(forecast_horizon),
        "var_names": list(var_names),
        "normalize": bool(normalize),
        "q_low": float(q_low),
        "q_high": float(q_high),
        "stats": _stats_to_jsonable(stats_vars, stats_params),
        "position_stats": asdict(pos_stats),
        "all_paths": all_paths,
        "splits": {
            "n_scenarios_total": len(all_paths),
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "train_paths": train_paths,
            "val_paths": val_paths,
            "test_paths": test_paths,
            "n_train_windows": len(ds_train),
            "n_val_windows": len(ds_val),
            "n_test_windows": len(ds_test),
            "train_frac": float(train_frac),
            "val_frac": float(val_frac),
            "seed": int(seed),
        },
        "graph": {
            "N": int(ds_train.N),
            "E": int(ds_train.edge_index.shape[0]),
            "pos_dim": 3,
            "position_normalization": "center by bounding-box midpoint, divide by max side length",
        },
    }

    if cache is None and split_stats_path is not None:
        _atomic_write_json(split_stats_path, info)

    return train_loader, val_loader, test_loader, info


# -----------------------------------------------------------------------------
# Full-trajectory helper for rollout evaluation/visualization
# -----------------------------------------------------------------------------

def load_normalized_trajectory(
    h5_path: str,
    var_names: Sequence[str],
    stats_vars: Dict[str, QuantileStats],
    normalize: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    with h5py.File(h5_path, "r") as f:
        tks = list_time_keys(f)
        if not tks:
            raise ValueError(f"{h5_path}: no time keys found")
        avail = list(f[tks[0]].keys())
        key_map = {vn: resolve_var_name(avail, vn) for vn in var_names}
        frames = []
        for tk in tks:
            g = f[tk]
            feats = []
            for vn in var_names:
                arr = g[key_map[vn]][:].astype(np.float32, copy=False)
                if normalize:
                    arr = stats_vars[vn].normalize(arr)
                feats.append(arr)
            frames.append(np.stack(feats, axis=-1))
    return np.stack(frames, axis=0).astype(np.float32), tks


def denormalize_channel(values: np.ndarray, info: Dict[str, object], channel: int) -> np.ndarray:
    var_name = info["var_names"][int(channel)]
    stats_vars, _ = stats_from_jsonable(info["stats"])
    return stats_vars[var_name].denormalize(values)
