"""Show paired four/eight label budgets and separate frozen count thresholds."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3bh'


def main():
    path=PUBLIC/'summary.json';s=json.loads(path.read_text());assert s['status']=='validated';rows=s['truth_results'];plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3bh-public'})
    fig,ax=plt.subplots(figsize=(14,11));fig.subplots_adjust(left=.34,right=.94,bottom=.24,top=.74)
    fig.suptitle('Eight queries pass the primary target; the stronger target still fails',x=.05,y=.96,ha='left',fontsize=17.5,fontweight='bold')
    fig.text(.05,.90,'PRIMARY SUPPORTED: all 22 truths ≥48/64. SECONDARY FALSIFIED: 19/22 truths ≥61/64.',fontsize=12,fontweight='bold')
    fig.text(.05,.835,'Same examined source banks and unchanged MAP-guided policy; only the label budget increases.\n1,371/1,408 exact after continuation: 667 additional successes, 37 unresolved cases retained.',fontsize=11.2,linespacing=1.5)
    labels=[]
    for i,r in enumerate(rows):
        labels.append(f"{r['candidate']}: "+' + '.join(f'({a},{b})' for a,b in r['orbits']));old=r['original_four_query_exact'];new=r['eight_query_exact'];ax.barh(i,old,height=.70,color='#426f94',label='Original ≤4 queries' if i==0 else None);ax.barh(i,new-old,left=old,height=.70,color='#72b29d',label='Additional successes with ≤8 total' if i==0 else None);ax.text(new+.65,i,str(new),va='center',fontsize=9.7,color='#a14d3f' if new<61 else '#276b56',bbox=dict(facecolor='white',edgecolor='none',pad=.4))
    ax.axvline(48,color='#555',ls='--',lw=1.2,zorder=0);ax.axvline(61,color='#a14d3f',ls=':',lw=1.5,zorder=0);ax.text(48,-1.1,'primary 48',ha='center',fontsize=9.6);ax.text(61,-1.1,'secondary 61',ha='center',fontsize=9.6,color='#a14d3f');ax.set_yticks(range(len(rows)),labels);ax.invert_yaxis();ax.set_xlim(0,69);ax.set_xticks([0,16,32,48,64]);ax.set_xlabel('Exact unique models among the same 64 source banks');ax.grid(axis='x',alpha=.14);ax.set_axisbelow(True);ax.legend(loc='upper center',bbox_to_anchor=(.4,1.115),frameon=False,fontsize=10,ncol=1)
    fig.text(.05,.17,'5,372 original binary labels + 1,308 additional = 6,680 cumulative. All 704 prior successes receive zero new labels.',fontsize=11.3,fontweight='bold')
    fig.text(.05,.11,'Candidates 1494, 1495 and 1537 remain below 61/64. Thirty-seven cases retain 2–7 candidates at the eight-query cap.\nBudget intervention on examined, selected hard-tail banks; no fresh confirmation or optimal-policy claim.',fontsize=11,linespacing=1.5)
    fig.text(.05,.045,f"All original 5,372 decisions replayed before continuation; new ledger references immutable parent records.\nBP AdamW: not measured. Study {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s. No new source draws.",fontsize=10.6,color='#444',linespacing=1.5)
    for ext in ['png','svg']:fig.savefig(PUBLIC/f'eight-query-continuation.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [path,Path(__file__)]},output_sha256={f'eight-query-continuation.{ext}':hashlib.sha256((PUBLIC/f'eight-query-continuation.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
