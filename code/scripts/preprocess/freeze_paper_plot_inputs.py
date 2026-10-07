"""Maintainer utility: freeze small plotting inputs from original experiment exports.

Not needed by the reproduction workflow. No checkpoints or raw trajectories
are copied. The source tree is the original final-paper-figures export layout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import h5py


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--pflotran-geometry', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('results/paper'))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    spatial = a.source/'three seed test rmse plots/spatial fields'
    records = json.loads((spatial/'final-final-plots/final_four_model_manifest_twig.json').read_text())
    si_record = json.loads((a.source/'final final plots/si_diffusion_top4_errors_blue_twig_manifest.json').read_text())
    si_models = si_record['source_models']
    with np.load(a.source/'si_diffusion_physical_data.npz') as b:
        np.savez_compressed(a.output/'si_paper_fields.npz', xy=b['xy'], edges=b['edges'],
            errors=np.stack([np.abs(b[m.replace(' ', '_')][-1, :, 0]-b['truth'][-1, :, 0]) for m in si_models]))
    air, pf = records['figures']
    inverse_labels = {display: source for source, display in records.get('display_label_mapping', {}).items()}
    air_models = [inverse_labels.get(m, m) for m in air['models']]
    pf_models = [inverse_labels.get(m, m) for m in pf['models']]
    with np.load(a.source/'airfoil_physical_data.npz') as b, np.load(spatial/'airfoil_mean_test_rmse_fields.npz') as e:
        np.savez_compressed(a.output/'airfoil_paper_fields.npz', xy=b['xy'], cells=b['cells'],
            errors=np.stack([e[m.replace(' ', '_')][:, 2:4] for m in air_models]))
    with np.load(a.source/'pflotran_physical_data.npz') as b, np.load(spatial/'pflotran_mean_test_rmse_fields.npz') as e:
        xyz=b['xyz'].astype(float)
        errors=np.stack([e[m.replace(' ', '_')][:, 0]*abs(float(b['scale'][0])) for m in pf_models])
    with h5py.File(a.pflotran_geometry) as h:
        raw=h['Domain/CoarseGraph/node_pos'][:]
        cells=h['Domain/Cells'][:]
        assert len(cells)%7==0 and np.all(cells[::7]==8)
        cells=cells.reshape(-1,7)[:,1:]
        top=cells[h['Domain/CoarseGraph/top_indices'][:]][:,3:6]
        vertices=h['Domain/Vertices'][:]
        bins=h['Domain/CoarseGraph/top_cell_bin'][:]
    np.testing.assert_allclose((raw-(raw.max(0)+raw.min(0))/2)/np.ptp(raw,axis=0).max(),xyz,atol=1e-6)
    span=np.ptp(xyz,axis=0)
    center=(raw.max(0)+raw.min(0))/2
    points=((vertices-center)/np.ptp(raw,axis=0).max()-(xyz.max(0)+xyz.min(0))/2)/span[1]
    points[:,2]*=.45*span[1]/span[2]
    az,el=np.deg2rad([-58,24])
    right=np.array([-np.sin(az),np.cos(az),0])
    up=np.array([-np.sin(el)*np.cos(az),-np.sin(el)*np.sin(az),np.cos(el)])
    depth=np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
    projected=np.column_stack((points@right,points@up))
    order=np.argsort((points[top]@depth).mean(1),kind='stable')
    polygons=projected[top[order]]
    all_points=polygons.reshape(-1,2);extent=np.ptp(all_points,axis=0)
    np.savez_compressed(a.output/'pflotran_paper_fields.npz', polygons=polygons, bins=bins[order], errors=errors,
        xlim=[all_points[:,0].min()-.04*extent[0],all_points[:,0].max()+.04*extent[0]],
        ylim=[all_points[:,1].min()-.04*extent[1],all_points[:,1].max()+.04*extent[1]])
    display_labels = si_record.get('display_label_mapping', {})
    si_record['source_models'] = [display_labels.get(m, m) for m in si_record['source_models']]
    si_record['display_label_mapping'] = {}
    metadata = {'si_diffusion': si_record, 'airfoil': air, 'pflotran': pf,
        'aggregation': records['aggregation'],
        'source_export': 'final-paper-figures',
        'geometry_source': a.pflotran_geometry.name,
        'geometry': records['surface_geometry'],
        'note': 'SI is selected-checkpoint absolute error; Airfoil/PFLOTRAN are mean per-run test RMSE. These are different statistics.'}
    # Local provenance paths are not runtime dependencies.
    metadata['geometry'].pop('source', None)
    (a.output/'paper_plot_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    inputs = ['si_paper_fields.npz', 'airfoil_paper_fields.npz', 'pflotran_paper_fields.npz',
              'paper_plot_metadata.json', 'pflotran_noise_summary.csv']
    checksums = {name: hashlib.sha256((a.output/name).read_bytes()).hexdigest() for name in inputs}
    (a.output/'plot_inputs_sha256.json').write_text(json.dumps(checksums, indent=2)+'\n')


if __name__ == '__main__':
    main()
