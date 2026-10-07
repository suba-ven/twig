import csv
from pathlib import Path

from twig.si_benchmark.direct14_config import Direct14Config
from twig.utils.model_registry import LEGACY_SI_MODELS


def test_si_direct14_protocol():
    cfg = Direct14Config()
    assert (cfg.history, cfg.forecast_horizon, cfg.rollout_steps) == (14, 14, 100)
    assert (cfg.n_train_scenarios, cfg.n_val_scenarios, cfg.n_test_scenarios) == (19, 3, 3)
    assert (cfg.epochs, cfg.batch_size, cfg.lr, cfg.early_stopping_patience) == (30, 8, 3e-4, 5)


def test_legacy_si_models_are_capacity_matched():
    path = Path(__file__).parents[1] / "results/si_diffusion/paper_summary/si_final_clean_rollout_summary.csv"
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    observed = {row["Model key"]: int(row["Parameters"]) for row in rows}
    assert set(LEGACY_SI_MODELS) == set(observed)
    assert observed["SA-TWIG-D14-K2"] == 70_224
    assert all(abs(parameters - 70_224) / 70_224 < 0.09 for parameters in observed.values())
