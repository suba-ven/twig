from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: tuple[int, ...] = (42, 43, 44)
    epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    target_parameters: int = 500_000
    output_root: Path = Path("results")

    def to_dict(self):
        data = asdict(self)
        data["output_root"] = str(self.output_root)
        return data
