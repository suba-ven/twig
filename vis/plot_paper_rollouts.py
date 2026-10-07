"""Figure 2: one rollout panel per dataset with consistent architecture colors."""
from pathlib import Path
import csv,json,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/codeocean/figures'
STEM='figure_2_rollouts_three_datasets'
PANELS=[('pflotran','a  PFLOTRAN',['TWIG','GPS Transformer','Graph WNO','GAT','MeshGraphNet']),('si_diffusion','b  SI diffusion',['TWIG','Graph FNO','GPS Transformer','Graph WNO','MeshGraphNet']),('airfoil','c  Airfoil',['GPS Transformer','TWIG','MeshGraphNet','Graph WNO','Graph FNO','GAT'])]
COLORS={'TWIG':'#0072B2','GPS Transformer':'#D55E00','Graph WNO':'#009E73','GAT':'#E69F00','MeshGraphNet':'#7D7D7D','Graph FNO':'#CC79A7'}
STYLES={'TWIG':'-','GPS Transformer':'--','Graph WNO':'-.','GAT':':','MeshGraphNet':(0,(3,1,1,1)),'Graph FNO':(0,(5,1))}

def main(zoom=False,twig=False,output=None):
    global OUT
    if output is not None: OUT=Path(output)
    OUT.mkdir(parents=True,exist_ok=True)
    selections=json.loads((ROOT/'configs/paper_selections.json').read_text(encoding="utf-8"))['datasets']
    stem=STEM+('_pflotran_zoom' if zoom else '')+('_twig' if twig else '')
    display=lambda label: label
    plt.rcParams.update({'font.family':'STIXGeneral','font.size':8,'axes.labelsize':8,'axes.titlesize':10,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(7.3,2.85),layout='constrained')
    manifest=dict(figure='Figure 2',aggregation='Arithmetic mean of three saved per-run test rollout RMSE curves; shaded band is sample SD (ddof=1), not a confidence interval.',colors=COLORS,panels=[])
    manifest['display_label_mapping']={m:display(m) for m in COLORS}
    export=[]
    for ax,(ds,title,labels) in zip(axes,PANELS):
        data=np.load(ROOT/'results/paper'/f'{ds}_curves.npz');record=dict(dataset=ds,models=[])
        for label in labels:
            curves=np.asarray(data[label],dtype=float)
            selected=selections[ds]['models'][label]
            assert curves.shape==(len(selected['seeds']),selections[ds]['rollout_steps']) and np.isfinite(curves).all()
            seeds=selected['seeds'];runs=selected.get('runs');source=selected.get('source', 'configs/paper_selections.json')
            mean=curves.mean(0);sd=curves.std(0,ddof=1);steps=np.arange(1,len(mean)+1)
            ax.fill_between(steps,np.maximum(0,mean-sd),mean+sd,color=COLORS[label],alpha=.10,linewidth=0,zorder=1)
            ax.plot(steps,mean,color=COLORS[label],linestyle=STYLES[label],linewidth=1.5 if label=='TWIG' else 1.25,zorder=3)
            record['models'].append(dict(model=label,seeds=seeds,runs=runs,sources=source,mean_final_step=float(mean[-1])))
            for j,seed in enumerate(seeds):
                for step,value in zip(steps,curves[j]):export.append([ds,label,seed,int(step),float(value)])
        ax.set_title(title,loc='left',fontweight='bold',pad=6)
        ax.set(xlabel='Rollout step',ylabel='Test RMSE',xlim=(1,len(mean)),ylim=(0,None))
        ax.set_xticks({'pflotran':[1,20,40,60],'si_diffusion':[1,25,50,75,100],'airfoil':[1,60,120,180]}[ds])
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18,linewidth=.5);ax.tick_params(length=3,width=.6)
        if ds=='pflotran' and zoom:
            detail=ax.inset_axes([.13,.59,.48,.32])
            low=[];high=[]
            for label in labels:
                avg=np.asarray(data[label],dtype=float).mean(0)[-5:]
                detail.plot(np.arange(56,61),avg,color=COLORS[label],linestyle=STYLES[label],linewidth=1.1)
                low.append(avg.min());high.append(avg.max())
            pad=(max(high)-min(low))*.12
            detail.set(xlim=(56,60),ylim=(min(low)-pad,max(high)+pad))
            detail.set_xticks([56,58,60]);detail.tick_params(labelsize=5.5,length=2,pad=1)
            detail.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
            detail.set_title('Steps 56–60 (means)',fontsize=6,pad=3)
            detail.grid(alpha=.15,linewidth=.4)
            for spine in detail.spines.values():spine.set_linewidth(.6);spine.set_color('0.4')
            record['zoom']=dict(steps=[56,60],ylim=list(detail.get_ylim()),display='Means only; full-panel shading retains seed SD')
        record['rollout_steps']=len(mean);manifest['panels'].append(record)
    handles=[Line2D([0],[0],color=COLORS[m],linestyle=STYLES[m],lw=1.5,label=display(m)) for m in COLORS]
    fig.legend(handles=handles,loc='outside upper center',ncol=3,frameon=False,fontsize=8,handlelength=2.5,columnspacing=2.1)
    for ext in ('png','pdf'):fig.savefig(OUT/(stem+'.'+ext),dpi=400,bbox_inches='tight')
    plt.close(fig)
    (OUT/(stem+'_manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    with (OUT/(stem+'_data.csv')).open('w') as fh:
        w=csv.writer(fh);w.writerow(['dataset','model','seed','rollout_step','test_rmse']);w.writerows(export)
    caption='Figure 2. Test rollout error on (a) PFLOTRAN, (b) SI diffusion, and (c) Airfoil. Lines show the mean over three runs; shaded bands show ±1 sample standard deviation. Architecture colors and line styles are consistent across datasets. Each panel has its own RMSE scale. PFLOTRAN and Airfoil use their saved channel-normalized metrics; SI uses infected-fraction RMSE. Airfoil GPS uses d6/h6 runs 01,03,05 (the three worst by mean rollout RMSE); d14 Graph FNO uses all three available runs. Other selections match the previous paper figures.'
    if zoom:caption+=' PFLOTRAN inset enlarges steps 56–60 with a truncated linear y-axis; inset lines show means only, while the main panel retains standard-deviation bands.'
    (OUT/(stem+'_caption.md')).write_text(caption+'\n')
    print('Saved',OUT/(stem+'.pdf'),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pflotran-zoom',action='store_true')
    parser.add_argument('--twig',action='store_true',help='Use TWIG in the legend and save a new figure copy.')
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();main(zoom=args.pflotran_zoom,twig=args.twig,output=args.output)
