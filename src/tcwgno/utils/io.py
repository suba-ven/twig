from __future__ import annotations
import csv, json
from dataclasses import asdict, is_dataclass
from pathlib import Path


def write_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(payload): payload = asdict(payload)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def write_history(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows: return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
