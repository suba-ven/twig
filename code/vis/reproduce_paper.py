"""Validate frozen manuscript inputs, regenerate figures, and export tables."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FIGURES = [
    'pflotran-noise-robustness-twig.png',
    'si_diffusion_top4_errors_blue_twig.png',
    'figure_2_rollouts_three_datasets_pflotran_zoom_twig.pdf',
    'pflotran_top4_final_pressure_rmse_pa_uncapped_gray_edges_twig.png',
    'airfoil_top4_final_compact_centered_zoom15_no_rmse_twig.png',
]


def validate_inputs(root=ROOT):
    reference = root/'results/paper'
    selections = json.loads((root/'configs/paper_selections.json').read_text())
    checksums = json.loads((reference/'plot_inputs_sha256.json').read_text())
    for dataset, selection in selections['datasets'].items():
        checksums[f'{dataset}_curves.npz'] = selection['curves_sha256']
    for name, expected in checksums.items():
        path = reference/name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen paper input checksum mismatch: {name}')
    metadata = json.loads((reference/'paper_plot_metadata.json').read_text())
    for dataset in selections['datasets']:
        models = metadata[dataset]['models']
        if any(m not in selections['datasets'][dataset]['models'] for m in models):
            raise ValueError(f'Spatial model selection mismatch: {dataset}')
    return checksums


def reproduce(output):
    output = Path(output).resolve()
    # Never let a typo overwrite versioned reference data or figures.
    for protected in (ROOT/'results/paper', ROOT/'figures', ROOT/'configs', ROOT/'src'):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError(f'Output overlaps source/reference files: {output}')
    checksums = validate_inputs()
    figures = output/'figures'
    tables = output/'tables'
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    def plot(script, *args):
        subprocess.run([sys.executable, str(ROOT/'vis'/script), *map(str,args)], check=True, cwd=ROOT)
    plot('plot_selected_results.py', '--output', output/'extended_data')
    plot('plot_selected_results.py', '--dataset', 'si_diffusion', '--ddof', 0, '--output', output/'si_table')
    plot('plot_paper_rollouts.py', '--pflotran-zoom', '--twig', '--output', figures)
    plot('plot_paper_noise.py', '--output', figures)
    plot('plot_paper_fields.py', '--output', figures)
    with (output/'extended_data/summary.csv').open() as f:
        rows = list(csv.DictReader(f))
    with (output/'si_table/summary.csv').open() as f:
        si_rows = {r['model']: r for r in csv.DictReader(f)}
    rows = [si_rows[r['model']] if r['dataset'] == 'si_diffusion' else r for r in rows]
    with (tables/'manuscript_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines = [r'\begin{tabular}{llrrr}', r'Dataset & Model & Parameters & Mean rollout RMSE & Final RMSE \\', r'\hline']
    for row in rows:
        mean = f"{float(row['mean_rollout_rmse']):.6g} $\\pm$ {float(row['std_rollout_rmse']):.3g}"
        final = f"{float(row['final_rmse']):.6g} $\\pm$ {float(row['std_final_rmse']):.3g}"
        lines.append(' & '.join([row['dataset'].replace('_',r'\_'),row['model'],row['parameters'],mean,final])+r' \\')
    (tables/'manuscript_summary.tex').write_text('\n'.join(lines+[r'\end{tabular}'])+'\n')
    shutil.copy2(ROOT/'results/paper/pflotran_noise_summary.csv', tables/'pflotran_noise_summary.csv')
    shutil.copy2(ROOT/'configs/paper_selections.json', output/'paper_selections.json')
    shutil.copy2(ROOT/'results/paper/paper_plot_metadata.json', output/'paper_plot_metadata.json')
    for name in FIGURES:
        if not (figures/name).is_file(): raise RuntimeError(f'Missing manuscript figure: {name}')
    report = {'workflow': 'frozen results; no training or inference', 'python': sys.version,
        'inputs_sha256': checksums, 'figures': FIGURES,
        'selections_sha256': hashlib.sha256((ROOT/'configs/paper_selections.json').read_bytes()).hexdigest(),
        'table_sd': {'pflotran': 1, 'si_diffusion': 0, 'airfoil': 1},
        'curve_band_sd': 1,
        'versions': {p: importlib.metadata.version(p) for p in ('numpy','matplotlib','pandas')},
        'outputs_sha256': {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in (figures,tables) for p in sorted(folder.iterdir()) if p.is_file()}}
    (output/'reproduction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Reproduced all five manuscript figures and 27 model/dataset table rows in {output}')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/codeocean')
    reproduce(parser.parse_args().output)
