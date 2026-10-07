#!/usr/bin/env python
"""Plot the frozen three-run results without the original cluster or checkpoints."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('all', 'pflotran', 'si_diffusion', 'airfoil'), default='all')
    p.add_argument('--output', type=Path, default=ROOT/'figures/selected')
    p.add_argument('--ddof', type=int, choices=(0, 1), default=1,
                   help='SD convention: 1 matches extended-data curves; 0 matches SI table/notebook 45')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT/'configs/paper_selections.json').read_text())['datasets']
    plt.rcParams.update({'font.family': 'STIXGeneral', 'font.size': 9, 'pdf.fonttype': 42})
    rows = []
    for dataset in (manifest if a.dataset == 'all' else [a.dataset]):
        selected = manifest[dataset]
        with np.load(ROOT/'results/paper'/f'{dataset}_curves.npz') as data:
            if set(data.files) != set(selected['models']): raise ValueError('Model selection mismatch')
            fig, ax = plt.subplots(figsize=(7.3, 3.3), layout='constrained')
            for label, entry in selected['models'].items():
                curves = data[label]
                if curves.shape != (len(entry['seeds']), selected['rollout_steps']) or not np.isfinite(curves).all():
                    raise ValueError(f'Invalid curves: {dataset}/{label}')
                mean, sd = curves.mean(0), curves.std(0, ddof=a.ddof)
                steps = np.arange(1, len(mean)+1)
                line, = ax.plot(steps, mean, label=label, lw=1.8 if label == 'TWIG' else 1.1)
                ax.fill_between(steps, mean-sd, mean+sd, color=line.get_color(), alpha=.1, lw=0)
                rows.append(dict(dataset=dataset, model=label, parameters=entry['parameters'],
                                 seeds='/'.join(map(str, entry['seeds'])), sd_ddof=a.ddof,
                                 mean_rollout_rmse=curves.mean(), std_rollout_rmse=curves.mean(1).std(ddof=a.ddof),
                                 final_rmse=curves[:, -1].mean(), std_final_rmse=curves[:, -1].std(ddof=a.ddof)))
            ax.set(xlabel='Rollout step', ylabel='RMSE')
            ax.spines[['top','right']].set_visible(False)
            ax.grid(alpha=.15)
            fig.legend(*ax.get_legend_handles_labels(), loc='outside upper center', ncol=5, frameon=False, fontsize=8)
            for ext in ('png','pdf'): fig.savefig(a.output/f'{dataset}_nine_models.{ext}', dpi=300, bbox_inches='tight')
            plt.close(fig)
    with (a.output/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == '__main__':
    main()
