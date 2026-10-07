"""Recreate the three manuscript spatial panels from frozen plotting inputs.

Layouts and scales follow the original final-paper-figures plotting scripts.
No inference, external datasets, or committed images are required.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.tri import Triangulation

ROOT = Path(__file__).resolve().parents[1]
BLUE = LinearSegmentedColormap.from_list('gray_blue',
    ['#d0d0d0', '#b9d4e8', '#79afd0', '#4292c6', '#084594'], N=1024)


def main(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((ROOT/'results/paper/paper_plot_metadata.json').read_text())
    plt.rcParams.update({'font.family': 'STIXGeneral', 'font.size': 8,
        'pdf.fonttype': 42, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7})
    for dataset, filename in [('si_diffusion', 'si'), ('airfoil', 'airfoil'), ('pflotran', 'pflotran')]:
        record = metadata[dataset]
        with np.load(ROOT/'results/paper'/f'{filename}_paper_fields.npz') as archive:
            b = dict(archive)
        labels = record['models']
        assert np.isfinite(b['errors']).all() and (b['errors'] >= 0).all()
        nr = 2 if dataset == 'airfoil' else 1
        fig = plt.figure(figsize=(7.3, 2.15 if nr == 2 else 2.6 if dataset == 'si_diffusion' else 2.0), layout='constrained')
        grid = fig.add_gridspec(nr, 5, width_ratios=[1,1,1,1,.045],
            wspace=.015, hspace=.015 if nr == 2 else .04)
        vmax = record['vmax'] if dataset == 'si_diffusion' else record['limits'][0]
        if dataset == 'airfoil':
            xy, cells = b['xy'], b['cells']
            triangles = Triangulation(xy[:,0], xy[:,1], cells)
            edges = np.sort(np.concatenate([cells[:,[0,1]],cells[:,[1,2]],cells[:,[2,0]]]),axis=1)
            unique, counts = np.unique(edges,axis=0,return_counts=True)
        for row in range(nr):
            for col, label in enumerate(labels):
                ax = fig.add_subplot(grid[row,col])
                values = b['errors'][col,:,row] if nr == 2 else b['errors'][col]
                if dataset == 'si_diffusion':
                    xy, edges = b['xy'], b['edges']
                    pairs = edges.T if edges.shape[0] == 2 else edges
                    ax.add_collection(LineCollection(xy[pairs],colors='0.83',linewidths=.15,zorder=0))
                    artist=ax.scatter(xy[:,0],xy[:,1],c=values,s=5,vmin=0,vmax=vmax,cmap=BLUE,linewidths=0,rasterized=True)
                elif dataset == 'airfoil':
                    artist=ax.tripcolor(triangles,values,shading='gouraud',vmin=0,vmax=vmax,cmap=BLUE,rasterized=True)
                    ax.add_collection(LineCollection(xy[unique[counts==1]],colors='0.55',linewidths=.35))
                    ax.add_collection(LineCollection(xy[unique],colors='#555555',linewidths=.12,alpha=.75,rasterized=True))
                    ax.set_xlim(record['xlim']); ax.set_ylim(record['ylim'])
                else:
                    artist=PolyCollection(b['polygons'],array=values[b['bins']],cmap=BLUE,
                        norm=Normalize(0,vmax),edgecolors='#a0a0a0',linewidths=.08,antialiased=True,rasterized=True)
                    ax.add_collection(artist);ax.set_xlim(b['xlim']);ax.set_ylim(b['ylim']);ax.set_axis_off()
                ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
                for spine in ax.spines.values(): spine.set_visible(False)
                if row == 0: ax.set_title(label,fontsize=9)
                if nr == 2 and col == 0: ax.set_ylabel(('x-velocity','y-velocity')[row],fontsize=9)
        units = {'si_diffusion':'Absolute infected-fraction error', 'airfoil':'Mean per-run normalized test RMSE', 'pflotran':'Pressure RMSE (Pa)'}[dataset]
        fig.colorbar(artist,cax=fig.add_subplot(grid[:,-1]),label=units)
        stem = 'si_diffusion_top4_errors_blue_twig' if dataset == 'si_diffusion' else record['stem']
        for ext in ('png','pdf'): fig.savefig(output/f'{stem}.{ext}',dpi=300,bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/codeocean/figures')
    main(parser.parse_args().output)
