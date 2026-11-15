"""Deterministic paired comparison from tracked H3am exact probabilities."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3am'


def main():
    path=PUBLIC/'summary.json';s=json.loads(path.read_text());assert s['status']=='validated';plt.rcParams.update({'font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3am-public'})
    fig,axes=plt.subplots(1,2,figsize=(14,7.5));fig.subplots_adjust(left=.075,right=.97,bottom=.35,top=.75,wspace=.32);fig.suptitle('D4 augmentation makes 32 positive acquisitions reliably sufficient',x=.065,y=.95,ha='left',fontsize=16.8,fontweight='bold');fig.text(.065,.879,'SUPPORTED for all 4 fixed fits: ≥90% reliability and ≥10 percentage-point paired improvement',fontsize=11.8,fontweight='bold');fig.text(.065,.819,'BP AdamW fits fixed · 32 raw positive labels per arm · D4 adds 256 attempted correlated views and a symmetry prior',fontsize=10.7)
    for bank,r in enumerate(s['rows']):
        for x,key,color in [(bank-.17,'raw_success_probability_pct','#7f8c8d'),(bank+.17,'augmented_success_probability_pct','#287f77')]:
            axes[0].bar(x,r[key],width=.32,color=color,label=('Unaugmented'if key.startswith('raw')else'D4 augmented')if bank==0 else None);axes[0].annotate(f"{r[key]:.2f}",(x,r[key]),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
        axes[1].bar(bank,r['gain_percentage_points'],color='#287f77',width=.62);axes[1].annotate(f"+{r['gain_percentage_points']:.2f} pp",(bank,r['gain_percentage_points']),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9.5)
    axes[0].axhline(90,color='#ae3f3a',ls='--',lw=1.3);axes[0].set_ylim(0,111);axes[0].set_ylabel('Exact P(capacity ≥95%) at 32 raw labels (%)');axes[0].set_title('A   Same raw draws, different positive evidence',loc='left',fontweight='bold',pad=14);axes[0].legend(frameon=False,loc='lower left',ncol=2,fontsize=9)
    axes[1].axhline(10,color='#ae3f3a',ls='--',lw=1.3);axes[1].text(-.3,11,'10 pp target',fontsize=9,color='#ae3f3a');axes[1].set_ylim(0,38);axes[1].set_ylabel('Paired reliability improvement (percentage points)');axes[1].set_title('B   Exact gain exceeds the frozen target',loc='left',fontweight='bold',pad=14)
    for ax in axes:ax.set_xticks(range(4),[f'Bank {b}'for b in range(4)]);ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
    fig.text(.065,.255,'D4 success probability is 96.044282% for every fit. Probability of raw success becoming D4 failure is exactly zero.',fontsize=11,fontweight='bold');fig.text(.065,.19,'D4 invariance is supplied as an additional prior. The views are deduplicated and correlated; they are not 256 independent labels.\nPure learner sees only its selected positives/model/classes. The evaluator alone scans the full 1,932-board source population.',fontsize=10.6,linespacing=1.5);fig.text(.065,.12,'Full joint raw/D4 union probabilities: all 1,792 paired masks retained, 32 distinct supports, independent grouped-DP check.\nSame seen 831,605-key N15 universe. Fixed-fit repair reliability; no refit, search-performance or fresh-generalization claim.',fontsize=10.6,linespacing=1.5);fig.text(.065,.05,f"Whole stage: {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s. Improvement is in raw-label use under the D4 prior.",fontsize=10.3,color='#444')
    for ext in('png','svg'):fig.savefig(PUBLIC/f'paired-D4-reliability.{ext}',dpi=180,facecolor='white',metadata={'Date':None}if ext=='svg'else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in(path,Path(__file__))},output_sha256={f'paired-D4-reliability.{ext}':hashlib.sha256((PUBLIC/f'paired-D4-reliability.{ext}').read_bytes()).hexdigest()for ext in('png','svg')});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
