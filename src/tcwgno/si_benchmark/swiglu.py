"""SwiGLU TC-WGNO variant used in the final SI-diffusion paper table."""
from dataclasses import fields, replace
import torch.nn as nn
import torch.nn.functional as F
from .direct14_registry import count_parameters, direct_sa_name

SWIGLU_NAME = "SwiGLU-TC-WGNO-D14-K5-H133"
SWIGLU_HIDDEN_MULT = 4.0 / 3.0


class SwiGLUChannelMixer(nn.Module):
    def __init__(self, width, hidden_mult=SWIGLU_HIDDEN_MULT, dropout=0.0):
        super().__init__()
        hidden = max(1, int(round(int(width) * float(hidden_mult))))
        self.in_proj = nn.Linear(int(width), 2 * hidden)
        self.out_proj = nn.Linear(hidden, int(width))
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

    def forward(self, x):
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(self.dropout(value * F.silu(gate)))


def replace_wavelet_ffns_with_swiglu(model, hidden_mult=SWIGLU_HIDDEN_MULT):
    replaced = []
    for name, module in model.named_modules():
        if not (hasattr(module, "wavelet") and hasattr(module, "ffn")):
            continue
        norm = getattr(module, "norm2", None)
        width = int(norm.normalized_shape[-1]) if isinstance(norm, nn.LayerNorm) else next(
            layer.in_features for layer in module.ffn.modules() if isinstance(layer, nn.Linear)
        )
        dropout = next((layer.p for layer in module.ffn.modules() if isinstance(layer, nn.Dropout)), 0.0)
        module.ffn = SwiGLUChannelMixer(width, hidden_mult, dropout)
        replaced.append(name)
    if not replaced:
        raise RuntimeError("No graph-wavelet FFN blocks were found")
    model.swiglu_replaced_modules = tuple(replaced)
    return model


def build_swiglu_spec(base_specs):
    base_name = direct_sa_name(5)
    base_spec = base_specs[base_name]
    factory = lambda: replace_wavelet_ffns_with_swiglu(base_spec.factory())
    probe = factory()
    metadata = dict(base_spec.metadata or {})
    metadata.update({"family": "TC-WGNO-SwiGLU", "channel_mixer": "SwiGLU",
                     "swiglu_hidden_mult": SWIGLU_HIDDEN_MULT, "matched_base": base_name})
    updates = {"name": SWIGLU_NAME, "factory": factory, "metadata": metadata,
               "parameters": count_parameters(probe), "label": "TC-WGNO"}
    valid = {field.name for field in fields(base_spec)}
    return replace(base_spec, **{key: value for key, value in updates.items() if key in valid})
