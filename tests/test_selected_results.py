"""Protect the frozen paper selections against protocol and curve drift."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SELECTIONS = json.loads((ROOT/'configs/paper_selections.json').read_text())['datasets']


@pytest.mark.parametrize('dataset,steps', [('pflotran',60),('si_diffusion',100),('airfoil',180)])
def test_frozen_curves(dataset, steps):
    selected = SELECTIONS[dataset]
    path = ROOT/'results/paper'/f'{dataset}_curves.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == selected['curves_sha256']
    with np.load(ROOT/'results/paper'/f'{dataset}_curves.npz') as data:
        assert len(data.files) == 9
        assert set(data.files) == set(selected['models'])
        for label in data.files:
            assert data[label].shape == (3, steps)
            assert np.isfinite(data[label]).all()
            assert (data[label] >= 0).all()
            assert len(selected['models'][label]['seeds']) == 3


def test_special_selections():
    si = SELECTIONS['si_diffusion']['models']['TWIG']
    assert si['seeds'] == [42,43,45]
    assert si['parameters'] == 70233
    air = SELECTIONS['airfoil']['models']
    assert air['GPS Transformer']['runs'] == [1,3,5]
    assert air['GPS Transformer']['seeds'] == [2026,2028,2030]
    assert air['Graph FNO']['architecture']['n_modes'] == 169
    assert air['Graph FNO']['parameters'] == 10001680
    assert SELECTIONS['pflotran']['models']['TWIG']['parameters'] == 1000620


def test_si_selected_swiglu_capacity():
    import torch
    from twig.si_benchmark.direct14_config import Direct14Config
    from twig.si_benchmark.direct14_registry import build_direct14_model_specs, count_parameters
    from twig.si_benchmark.graph import make_graph_static
    from twig.si_benchmark.graph_wno_reference import GraphWNOBlock3D
    from twig.si_benchmark.swiglu import build_swiglu_spec
    n = 70
    graph = make_graph_static(torch.zeros(n,42), torch.zeros(n,2),
                              torch.stack([torch.arange(n),torch.arange(n).roll(1)]), torch.ones(n))
    specs,_ = build_direct14_model_specs(Direct14Config(), graph, torch.eye(n)[:,:64],
                                        torch.arange(64).float(), GraphWNOBlock3D,
                                        torch.device('cpu'), sa_band_counts=(2,5))
    model = build_swiglu_spec(specs).factory()
    assert count_parameters(model) == 70233


def test_current_registry_matches_selections():
    from twig.utils.model_registry import PFLOTRAN_MODELS, SI_MODELS
    for dataset, names in [('pflotran', PFLOTRAN_MODELS), ('si_diffusion', SI_MODELS)]:
        assert set(names) == {entry['model_key'] for entry in SELECTIONS[dataset]['models'].values()}


def test_airfoil_rejects_stride3_with_same_number_of_states(tmp_path):
    from twig.airfoil_benchmark.data import load_meta
    (tmp_path/'meta.json').write_text(json.dumps({'trajectory_length':200, 'temporal_subsampling':{'stride':3}}))
    with pytest.raises(ValueError, match='contiguous'):
        load_meta(tmp_path)
