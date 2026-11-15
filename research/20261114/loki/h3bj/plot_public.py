"""Show fresh paired accuracy and actual per-policy label-cost differences."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3bj'


def main():
    path=PUBLIC/'summary.json';s=json.loads(path.read_text());assert s['status']=='validated';rows=s['truth_results'];plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3bj-public'})
    fig,(ax,cost)=plt.subplots(1,2,figsize=(14,11),gridspec_kw={'width_ratios':[2.1,1]});fig.subplots_adjust(left=.13,right=.94,bottom=.26,top=.74,wspace=.30)
    fig.suptitle('Fresh banks: exact-tree continuation resolves every tested model',x=.045,y=.96,ha='left',fontsize=19,fontweight='bold');fig.text(.045,.90,'A SUPPORTED: all 22 truths reach 64/64. B SUPPORTED: 37 hybrid-only successes, zero MAP-only.',fontsize=12,fontweight='bold');fig.text(.045,.835,'1,408 fresh random draws on the selected hard tail; both policies share the first four MAP queries.\nSecondary supported: all 701 unresolved checkpoints admit four-query trees identifying every remaining candidate.',fontsize=11.4,linespacing=1.5)
    for i,r in enumerate(rows):
        base=r['baseline_exact'];extra=r['hybrid_exact']-base;ax.barh(i,base,color='#517d9b',height=.7,label='Unchanged ≤8 MAP' if i==0 else None);ax.barh(i,extra,left=base,color='#65a78f',height=.7,label='Additional hybrid successes' if i==0 else None)
        if extra:ax.text(base-.65,i,str(base),ha='right',va='center',color='white',fontsize=9.7)
        delta=r['hybrid_policy_label_cost']-r['baseline_policy_label_cost'];cost.barh(i,delta,color='#a86951' if delta>0 else '#54947c',height=.7);cost.text(delta+(1.2 if delta>=0 else -1.2),i,f'{delta:+d}' if delta else '0',ha='left' if delta>=0 else 'right',va='center',fontsize=9.8)
    ax.set_yticks(range(len(rows)),[str(r['candidate']) for r in rows]);ax.set_ylabel('True candidate ID');ax.invert_yaxis();ax.set_xlim(0,68);ax.set_xticks([0,16,32,48,64]);ax.set_xlabel('Exact cases among 64 source banks');ax.axvline(61,color='#555',ls=':',lw=1,zorder=0);ax.legend(loc='upper left',bbox_to_anchor=(0,1.10),frameon=False,fontsize=10,ncol=2);cost.set_yticks([]);cost.invert_yaxis();cost.axvline(0,color='#666',lw=.8);cost.set_xlim(-26,43);cost.set_title('Hybrid minus MAP label cost',fontsize=11,pad=20);cost.set_xlabel('Total binary labels across 64 banks')
    for a in [ax,cost]:a.grid(axis='x',alpha=.14);a.set_axisbelow(True)
    fig.text(.045,.195,'Hybrid: 1,408/1,408 exact with 6,666 policy labels. MAP: 1,371/1,408 exact with 6,647 labels.',fontsize=12,fontweight='bold');fig.text(.045,.13,'Accuracy improves at a net cost of 19 more labels; no minimum-depth or lower-cost claim.\n5,339 shared + 1,327 hybrid + 1,308 MAP = 7,974 actual calls; 691 cross-branch repeats are explicitly charged.',fontsize=11.1,linespacing=1.5);fig.text(.045,.055,f"22,528 positive-example appearances counted separately. No exact bank duplicates observed; individual boards may overlap.\nBP AdamW: not measured. Study {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s. Selected finite class, no significance claim.",fontsize=10.7,color='#444',linespacing=1.5)
    for ext in ['png','svg']:fig.savefig(PUBLIC/f'fresh-paired-policy.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [path,Path(__file__)]},output_sha256={f'fresh-paired-policy.{ext}':hashlib.sha256((PUBLIC/f'fresh-paired-policy.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
