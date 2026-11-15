"""Plot exact conditional dose curves, retaining every decrease and archived point."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3aw'


def main():
    path=PUBLIC/'summary.json';s=json.loads(path.read_text());assert s['status']=='validated'
    plt.rcParams.update({'font.size':10.8,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3aw-public'})
    fig,axes=plt.subplots(1,2,figsize=(14,8.2));fig.subplots_adjust(left=.075,right=.965,bottom=.36,top=.75,wspace=.3)
    fig.suptitle('More positives eventually pass 99%, with measured intermediate decreases',x=.065,y=.95,ha='left',fontsize=16.5,fontweight='bold')
    fig.text(.065,.88,'SUPPORTED at 128 total labels: every bank exceeds 99% conditional probability of safe ≥95%-capacity MAP',fontsize=11.4,fontweight='bold')
    fig.text(.065,.818,'Unchanged hierarchical prior; exact hypothetical uniform acquisition from each fixed initial 16-label complement',fontsize=10.7)
    specifications=[(0,[(0,'Z0','#b06432','o'),(1,'Z1 = Z2','#3776a3','s'),(3,'Z3','#4f8b5e','^')]),(1,[(4,'L0 = L3','#747474','s'),(5,'L1','#267b72','o'),(6,'L2','#8657a5','^')])]
    for family,series in specifications:
        ax=axes[family]
        for index,label,color,marker in series:
            rows=[r for r in s['rows'] if r['index']==index];doses=[r['total_labels'] for r in rows];prob=[r['success_pct'] for r in rows];ax.plot(doses,prob,color=color,lw=1.8,label=label)
            ax.scatter(doses[:2],prob[:2],s=55,marker=marker,facecolors='white',edgecolors=color,zorder=4)
            ax.scatter(doses[2:],prob[2:],s=55,marker=marker,color=color,zorder=4)
        ax.set_xscale('log',base=2);ax.set_xticks([16,32,64,128],[16,32,64,128]);ax.set_xlabel('Total positive labels (initial 16 + hypothetical additions)');ax.axhline(99,color='#b54440',ls='--',lw=1.1);ax.grid(axis='y',alpha=.16);ax.legend(frameon=False,loc='lower right',fontsize=9.5)
    axes[0].set_ylim(78,103);axes[0].set_ylabel('P(all MAP ties safe and ≥95% capacity), %');axes[0].set_title('A   Zebra: zoomed 78–103% scale exposes regressions',loc='left',fontsize=11.2,fontweight='bold',pad=15)
    axes[1].set_ylim(-5,105);axes[1].set_ylabel('P(all MAP ties safe and ≥95% capacity), %');axes[1].set_title('B   Leaper: failed L2 improves from 0% to >99.99%',loc='left',fontsize=11.2,fontweight='bold',pad=15)
    axes[0].annotate('Z0: 99.872% → 99.690%',xy=(128,99.6903062843),xytext=(40,85.4),arrowprops=dict(arrowstyle='->',color='#b06432'),color='#935226',fontsize=9.3)
    axes[0].annotate('Z1/Z2: 100% → 97.512%',xy=(64,97.5122490834),xytext=(18,81.4),arrowprops=dict(arrowstyle='->',color='#3776a3'),color='#3776a3',fontsize=9.3)
    fig.text(.065,.275,'Open markers: archived 16 / 32 outcomes, exactly revalidated. Filled markers: prospective 64 / 128 outcomes.',fontsize=10.6,fontweight='bold')
    fig.text(.065,.215,'At 128 labels, success probabilities are 99.682%–100%; Z0 still declines from 64 to 128.\nA newly competitive false (9,11) orbit accounts for the late zebra regression. All false models and failed events remain recorded.',fontsize=10.7,linespacing=1.5)
    fig.text(.065,.14,'L0 and L3 are exactly 100% at every dose. The other six banks retain nonzero failure probability at 128.\nThese curves are conditional marginal probabilities at four fixed doses; connecting points does not describe every acquisition path.',fontsize=10.5,linespacing=1.5)
    fig.text(.065,.06,f"No new observed labels consumed; same seen initial banks, source families and N13 targets. No prior tuning or extra dose search.\nAll 416 mask-dose cases and 28 complete supports retained. Study: {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s.",fontsize=10.3,color='#444',linespacing=1.5)
    for ext in ('png','svg'):fig.savefig(PUBLIC/f'conditional-dose-curve.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (path,Path(__file__))},output_sha256={f'conditional-dose-curve.{ext}':hashlib.sha256((PUBLIC/f'conditional-dose-curve.{ext}').read_bytes()).hexdigest() for ext in ('png','svg')});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
