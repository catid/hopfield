"""Show concrete bidirectional witnesses and exact random-positive witness rates."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3ay'


def main():
    paths=[PUBLIC/'summary.json',PUBLIC/'distinguishing-witnesses.json'];s=json.loads(paths[0].read_text());w=json.loads(paths[1].read_text())['witnesses'];assert s['status']=='validated' and len(w)==2
    plt.rcParams.update({'font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3ay-public'})
    fig,axes=plt.subplots(1,3,figsize=(15,8),gridspec_kw={'width_ratios':[1.15,1.15,.8]});fig.subplots_adjust(left=.065,right=.97,bottom=.3,top=.74,wspace=.27)
    fig.suptitle('N13 breaks the source alias, but sixteen positives rarely provide a direct witness',x=.06,y=.95,ha='left',fontsize=16.5,fontweight='bold')
    fig.text(.06,.88,'FALSIFIED: both directed differences exist, but direct-witness probability is only 19.62% for A and 65.46% for B',fontsize=11.4,fontweight='bold')
    fig.text(.06,.818,'N12: identical 120-board populations. N13: A-only 16 boards · shared 1,172 · B-only 80 boards.',fontsize=11)
    checker=np.indices((13,13)).sum(0)%2
    for ax,record in zip(axes[:2],w):
        ax.imshow(checker,cmap=ListedColormap(['#f3eee3','#d9dfda']),vmin=0,vmax=1);cols=record['columns_zero_based'];ax.scatter(cols,range(13),s=62,color='#263c39',zorder=3)
        for pair in record['all_violating_occupied_pairs_under_other']:
            ax.plot([pair['col_a'],pair['col_b']],[pair['row_a'],pair['row_b']],color='#b6443c',lw=2,zorder=4);ax.scatter([pair['col_a'],pair['col_b']],[pair['row_a'],pair['row_b']],s=82,color='#b6443c',zorder=5)
        ax.set_xticks(range(13),range(1,14),fontsize=7.6);ax.set_yticks(range(13),range(1,14),fontsize=7.6);ax.tick_params(length=0);ax.set_xlabel('Column (one-based)');ax.set_ylabel('Row (one-based)');ax.set_title(f"Valid under {record['valid_under']}; invalid under {record['invalid_under']}",loc='left',fontweight='bold',fontsize=11.6,pad=14)
    axes[0].text(.5,-.19,'Violates B’s (2,7) rule only',transform=axes[0].transAxes,ha='center',fontsize=10,color='#a13f38');axes[1].text(.5,-.19,'Violates A’s (3,4) rule only',transform=axes[1].transAxes,ha='center',fontsize=10,color='#a13f38')
    for i,row in enumerate(s['rows']):axes[2].bar(i,row['distinguishing_pct'],.58,color=['#287c70','#4a769b'][i]);axes[2].text(i,row['distinguishing_pct']+3,f"{row['distinguishing_pct']:.2f}%",ha='center',fontsize=10.5)
    axes[2].axhline(95,color='#b6443c',ls='--',lw=1.2);axes[2].set_ylim(0,108);axes[2].set_xticks([0,1],['A true','B true']);axes[2].set_ylabel('P(≥1 direct witness in 16 distinct positives), %');axes[2].set_title('Uniform positive sampling',loc='left',fontweight='bold',fontsize=11.2,pad=14);axes[2].grid(axis='y',alpha=.15);axes[2].set_axisbelow(True)
    fig.text(.06,.19,'A = queens + {(1,2), (3,4)}.  B = queens + {(1,2), (2,7)}.  Red pairs violate the other model.',fontsize=10.9,fontweight='bold')
    fig.text(.06,.125,'Witnesses are the lexicographically first complete boards in each directed difference. All sets and violation pairs are public.\nThe event is logical rejection of the other model by a positive board; it is not MAP selection accuracy or identification among 1,541 models.',fontsize=10.6,linespacing=1.5)
    fig.text(.06,.05,f"Exact failure = C(1,172,16) / C(true population size,16). No new observed labels, prior tuning or negative-query supervision.\nComplete populations independently enumerated; full clue censuses retained. Study: {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s.",fontsize=10.2,color='#444',linespacing=1.5)
    for ext in ('png','svg'):fig.savefig(PUBLIC/f'alias-witnesses.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[Path(__file__)]},output_sha256={f'alias-witnesses.{ext}':hashlib.sha256((PUBLIC/f'alias-witnesses.{ext}').read_bytes()).hexdigest() for ext in ('png','svg')});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
