"""Factories for the nine models in the final 1M-parameter paper comparison."""
from .pflotran_h10_models import build_pflotran_h10_model_specs
from .pflotran_h10_wavelet_ffn_alternatives import build_wavelet_swiglu_spec


def build_paper_specs(graph, modes, eigenvalues):
    specs, matched = build_pflotran_h10_model_specs(
        graph, modes, eigenvalues, history=10, forecast_horizon=10, channels=2,
        target_parameters=1_000_000, sa_band_counts=(), operator_depth=10,
        message_depth=10, attention_depth=3, n_modes=128, dropout=0.05,
    )
    tc, tc_config = build_wavelet_swiglu_spec(
        graph, modes, eigenvalues, bands=5, n_modes=128, operator_depth=10,
        encoder_width=512, encoder_depth=2, swiglu_mult=4/3, n_scales=4,
        dropout=0.05, target_parameters=1_000_000, width_candidates=(120,),
    )
    specs[tc.name] = tc
    matched[tc.name] = tc_config
    return specs, matched
