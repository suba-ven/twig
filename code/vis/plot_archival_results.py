"""Regenerate the retained historical comparisons with TWIG branding.

These use historical frozen curves, not the manuscript's current selections.
The layouts follow the original PFLOTRAN and SI benchmark notebooks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]


def save(fig, output, stem):
    for ext in ('pdf', 'png'):
        fig.savefig(output/f'{stem}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def load(dataset):
    folder = ROOT/'results'/dataset/'paper_summary'
    path = folder/'archival_curves.npz'
    expected = json.loads((folder/'archival_curves.json').read_text())['sha256']
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'Archival curve checksum mismatch: {dataset}')
    with np.load(path) as data:
        return dict(data)


def pflotran(output):
    data = load('pflotran')
    all_models = ['TWIG', 'Graph FNO', 'Plain Graph WNO', 'MeshGraphNet', 'GAT', 'GATv2',
                  'TWIG (K=3)', 'GPS Transformer', 'RNN-GNN Fusion', 'RNN']
    groups = [
        ('ten_model_three_seed_rollout_curves_step60', all_models, 'PFLOTRAN 60-step rollout: all models', 'linear'),
        ('ten_model_three_seed_rollout_curves_step60_log', all_models, 'PFLOTRAN 60-step rollout: all models', 'log'),
        ('ten_model_competitive_zoom_step60', [m for m in all_models if m != 'Graph FNO'], 'PFLOTRAN 60-step rollout: competitive models', 'linear'),
        ('operator_comparison_step60', ['TWIG','TWIG (K=3)','Plain Graph WNO','Graph FNO'], 'Graph neural operator comparison', 'linear'),
        ('operator_comparison_step60_log', ['TWIG','TWIG (K=3)','Plain Graph WNO','Graph FNO'], 'Graph neural operator comparison', 'log'),
        ('competitive_operator_comparison_step60', ['TWIG','TWIG (K=3)','Plain Graph WNO'], 'Competitive graph neural operators', 'linear'),
        ('graph_network_comparison_step60', ['TWIG','GPS Transformer','GAT','GATv2','MeshGraphNet'], 'Graph network and transformer comparison', 'linear'),
        ('recurrent_comparison_step60', ['TWIG','GPS Transformer','RNN-GNN Fusion','RNN'], 'Recurrent-model comparison', 'linear'),
        ('top_methods_comparison_step60', ['TWIG','TWIG (K=3)','GPS Transformer','GAT','MeshGraphNet','Plain Graph WNO','GATv2'], 'PFLOTRAN 60-step rollout', 'linear'),
    ]
    plt.rcParams.update({'font.family':'serif', 'mathtext.fontset':'cm', 'font.size':9,
        'axes.labelsize':9, 'axes.titlesize':10, 'legend.fontsize':7.5, 'xtick.labelsize':8,
        'ytick.labelsize':8, 'axes.linewidth':.8, 'pdf.fonttype':42, 'ps.fonttype':42})
    for stem, labels, title, scale in groups:
        fig, ax = plt.subplots(figsize=(8.8 if len(labels)>=8 else 7.1, 3.8))
        steps=np.arange(1,61)
        for index, label in enumerate(labels):
            curves=np.asarray(data[label], dtype=float)
            mean, sd=curves.mean(0), curves.std(0,ddof=0)
            line,=ax.plot(steps,mean,label=label,linewidth=1.8,linestyle=['-','--','-.',':'][index%4])
            ax.fill_between(steps,np.maximum(mean-sd,1e-12),mean+sd,alpha=.10,color=line.get_color(),linewidth=0)
        for boundary in range(10,60,10):ax.axvline(boundary+.5,linestyle=':',linewidth=.6,alpha=.18,zorder=0)
        ax.set(xlabel='Autoregressive forecast step',ylabel='RMSE',xlim=(1,60),yscale=scale,title=title)
        ax.set_xticks([1,10,20,30,40,50,60]);ax.grid(True,linewidth=.4,alpha=.30)
        if len(labels)>=8:
            ax.legend(loc='center left',bbox_to_anchor=(1.01,.5),frameon=True,framealpha=.92,borderpad=.4,handlelength=2.5)
            fig.tight_layout(rect=(0,0,.78,1))
        else:
            ax.legend(loc='upper left',ncol=2 if len(labels)>=4 else 1,frameon=True,framealpha=.92,borderpad=.4,handlelength=2.5)
            fig.tight_layout()
        save(fig,output,stem)


def si_diffusion(output):
    data=load('si_diffusion')
    labels={'SwiGLU-TWIG-D14-K6':'TWIG','SA-TWIG-D14-K6':'TWIG (standard FFN)',
        'SA-TWIG-D14-K2':r'TWIG ($K=2$)', 'GPS-Transformer-D14':'GPS Transformer',
        'GAT-D14':'GAT','MeshGraphNet-D14':'MeshGraphNet','GWNO-D14':'Graph WNO',
        'GATv2-D14':'GATv2','RNN-GNN-Fusion-D14':'RNN-GNN Fusion','RNN-D14':'RNN','GraphFNO-D14':'Graph FNO'}
    names=list(labels)
    if set(names)!=set(data): raise ValueError(f'Unexpected archived models: {set(data)^set(names)}')
    competitive=[names[0]]+sorted(names[1:],key=lambda n:data[n].mean())[:6]
    colors={n:plt.get_cmap('tab20')(i) for i,n in enumerate(names)}
    markers=dict(zip(names,['o','s','D','v','^','P','X','<','>','*','h']))
    styles=dict(zip(names,['-','--',':',(0,(5,2)),'-.',(0,(3,1,1,1)),(0,(1,1)),(0,(6,2,1,2)),(0,(2,2)),'-',(0,(4,1))]))
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':8.3,
        'axes.labelsize':8.7,'axes.titlesize':9,'legend.fontsize':6.6,'xtick.labelsize':7.7,
        'ytick.labelsize':7.7,'axes.linewidth':.75,'lines.linewidth':1.55,'pdf.fonttype':42,'ps.fonttype':42})
    def panel(ax, group, letter=None):
        for name in group:
            curves=np.asarray(data[name],dtype=float)[:,:100];mean=curves.mean(0);sd=curves.std(0,ddof=0)
            ax.plot(np.arange(1,101),mean,color=colors[name],linestyle=styles[name],marker=markers[name],
                markevery=20,markersize=3.1,markeredgewidth=.4,linewidth=2.2 if name==names[0] else 1.3,
                label=labels[name],zorder=5 if name==names[0] else 2)
            ax.fill_between(np.arange(1,101),np.maximum(mean-sd,0),mean+sd,color=colors[name],alpha=.075,linewidth=0,zorder=0)
        for boundary in range(14,100,14):ax.axvline(boundary+.5,linestyle=':',linewidth=.45,alpha=.13,zorder=0)
        ax.set(xlim=(1,100),xlabel='Forecast step',ylabel='RMSE');ax.set_xticks([1,25,50,75,100])
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',linewidth=.42,alpha=.25)
        if letter:ax.text(.02,.97,letter,transform=ax.transAxes,ha='left',va='top',fontweight='bold',fontsize=8.8)
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.05))
    panel(axes[0],names,'(a)');panel(axes[1],competitive,'(b)')
    handles=[Line2D([0],[0],color=colors[n],marker=markers[n],linestyle=styles[n],linewidth=2.1 if n==names[0] else 1.3,markersize=3.2,label=labels[n]) for n in names]
    fig.legend(handles=handles,labels=[labels[n] for n in names],loc='upper center',bbox_to_anchor=(.5,1.025),ncol=6,frameon=False,columnspacing=.65,handlelength=1.9,handletextpad=.32)
    fig.subplots_adjust(left=.085,right=.995,bottom=.18,top=.73,wspace=.27)
    save(fig,output,'si-rollout-all-and-competitive')
    for group,stem in [(names,'si-rollout-all-models'),(competitive,'si-rollout-competitive-models')]:
        fig,ax=plt.subplots(figsize=(4.8,3.25));panel(ax,group)
        ax.legend(frameon=False,ncol=2,columnspacing=.65,handlelength=1.9);fig.tight_layout();save(fig,output,stem)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/codeocean/archival_figures')
    args=parser.parse_args()
    for dataset,render in [('pflotran',pflotran),('si_diffusion',si_diffusion)]:
        output=args.output/dataset;output.mkdir(parents=True,exist_ok=True);render(output)
