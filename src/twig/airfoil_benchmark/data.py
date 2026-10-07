from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tfrecord.torch.dataset import TFRecordDataset


DYNAMIC_FIELDS = ("density", "pressure", "velocity")


def load_meta(data_dir: Path) -> dict:
    meta = json.loads((Path(data_dir) / "meta.json").read_text())
    stride = meta.get("temporal_subsampling", {}).get("stride", 1)
    if meta.get("trajectory_length") != 200 or stride != 1:
        raise ValueError("The selected Airfoil protocol requires the original contiguous 200 states, not temporal subsamples")
    return meta


def description(meta: dict) -> dict[str, str]:
    return {name: "byte" for name in meta["field_names"]}


def feature_bytes(record: dict, key: str) -> bytes:
    value = record[key]
    if isinstance(value, bytes):
        return value
    value = np.asarray(value).reshape(-1)
    if value.size != 1 or not isinstance(value[0], bytes):
        raise TypeError(f"Unexpected TFRecord value for {key}: {type(record[key])}")
    return value[0]


def decode_array(record: dict, meta: dict, key: str) -> np.ndarray:
    spec = meta["features"][key]
    return np.frombuffer(feature_bytes(record, key), dtype=np.dtype(spec["dtype"])).reshape(spec["shape"])


def decode_state(record: dict, meta: dict) -> np.ndarray:
    return np.concatenate([decode_array(record, meta, key) for key in DYNAMIC_FIELDS], axis=-1).astype(np.float32, copy=False)


def iter_records(data_dir: Path, split: str):
    meta = load_meta(data_dir)
    yield from TFRecordDataset(
        str(Path(data_dir) / f"{split}.tfrecord"), None, description=description(meta)
    )


def compute_training_stats(data_dir: Path) -> dict:
    meta = load_meta(data_dir)
    total = np.zeros(4, dtype=np.float64)
    total_sq = np.zeros(4, dtype=np.float64)
    count = 0
    records = 0
    for record in iter_records(data_dir, "train"):
        state = decode_state(record, meta).astype(np.float64)
        total += state.sum(axis=(0, 1))
        total_sq += np.square(state).sum(axis=(0, 1))
        count += state.shape[0] * state.shape[1]
        records += 1
        if records % 100 == 0:
            print(f"stats: {records} trajectories", flush=True)
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1.0e-12)
    return {"mean": mean.tolist(), "std": np.sqrt(variance).tolist(), "count": count, "trajectories": records}


def extract_static_graph(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    meta = load_meta(data_dir)
    record = next(iter(iter_records(data_dir, "train")))
    pos = decode_array(record, meta, "mesh_pos")[0].astype(np.float32)
    cells = decode_array(record, meta, "cells")[0].astype(np.int64)
    node_type = decode_array(record, meta, "node_type")[0].astype(np.int64)
    directed = set()
    for tri in cells:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            directed.add((int(a), int(b)))
            directed.add((int(b), int(a)))
    edge_index = np.asarray(sorted(directed), dtype=np.int64).T
    return pos, edge_index, node_type


class AirfoilWindowDataset(IterableDataset):
    def __init__(self, data_dir: Path, split: str, stats: dict, history: int, horizon: int,
                 windows_per_trajectory: int, seed: int, random_windows: bool,
                 channel_indices: tuple[int, ...] | None = None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.meta = load_meta(data_dir)
        self.channel_indices = channel_indices
        self.mean = np.asarray(stats["mean"], dtype=np.float32)
        self.std = np.asarray(stats["std"], dtype=np.float32)
        if channel_indices is not None:
            indices = np.asarray(channel_indices, dtype=np.int64)
            self.mean = self.mean[indices]
            self.std = self.std[indices]
        self.history = int(history)
        self.horizon = int(horizon)
        self.windows = int(windows_per_trajectory)
        self.seed = int(seed)
        self.random_windows = bool(random_windows)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        workers = 1 if worker is None else worker.num_workers
        rng = random.Random(self.seed + 1009 * self.epoch + worker_id)
        span = self.history + self.horizon
        for trajectory_index, record in enumerate(iter_records(self.data_dir, self.split)):
            if trajectory_index % workers != worker_id:
                continue
            state = decode_state(record, self.meta)
            if self.channel_indices is not None:
                state = state[..., self.channel_indices]
            state = (state - self.mean) / self.std
            max_start = state.shape[0] - span
            if max_start < 0:
                raise ValueError(f"Trajectory has {state.shape[0]} frames, need {span}")
            if self.random_windows:
                starts = [rng.randint(0, max_start) for _ in range(self.windows)]
            else:
                starts = np.linspace(0, max_start, self.windows, dtype=int).tolist()
            for start in starts:
                x = np.ascontiguousarray(state[start:start + self.history])
                y = np.ascontiguousarray(state[start + self.history:start + span])
                yield torch.from_numpy(x), torch.from_numpy(y)


def make_loader(dataset: AirfoilWindowDataset, batch_size: int, workers: int = 0) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, num_workers=workers, pin_memory=True,
                      persistent_workers=False)
