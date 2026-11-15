"""Show known observed successes and new identifiable failures as distinct evidence."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3bg'


def main():
    path=PUBLIC/'summary.json';s=json.loads(path.read_text());assert s['status']=='validated';rows=s['truth_results'];plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3bg-public'})
    fig,ax=plt.subplots(figsize=(14,11));fig.subplots_adjust(left=.34,right=.94,bottom=.24,top=.74)
    fig.suptitle('All 704 unresolved cases are identifiable in principle at N13',x=.05,y=.96,ha='left',fontsize=18,fontweight='bold')
    fig.text(.05,.90,'SUPPORTED: every initial candidate set has a singleton intersection with the true N13 prediction class.',fontsize=12,fontweight='bold')
    fig.text(.05,.835,'Zero new labels. The lighter segments are exact identifiability diagnostics, not newly successful query runs.\nThis rules out unavoidable N13 ambiguity; it supplies no four-query or optimal-query guarantee.',fontsize=11.2,linespacing=1.5)
    labels=[]
    for i,r in enumerate(rows):
        labels.append(f"{r['candidate']}: "+' + '.join(f'({a},{b})' for a,b in r['orbits']));known=r['observed_four_query_exact'];possible=r['failed_cases_identifiable'];ax.barh(i,known,height=.70,color='#426f94',label='Observed four-query successes (704)' if i==0 else None);ax.barh(i,possible,left=known,height=.70,color='#bbddd3',hatch='///',edgecolor='#7db4a3',label='Prior failures, identifiable in principle (704)' if i==0 else None);ax.text(64.7,i,'64',va='center',fontsize=9.6,color='#426d5e')
    ax.set_yticks(range(len(rows)),labels);ax.invert_yaxis();ax.set_xlim(0,69);ax.set_xticks([0,16,32,48,64]);ax.set_xlabel('Cases per truth: observed successes plus distinguishable prior failures');ax.grid(axis='x',alpha=.14);ax.set_axisbelow(True);ax.legend(loc='upper center',bbox_to_anchor=(.40,1.09),frameon=False,fontsize=9.8,ncol=1)
    fig.text(.05,.17,'Every truth has ceiling 64/64. All 1,703 truth/initial-competitor pairs retain both directed differences and witnesses.',fontsize=11.3,fontweight='bold')
    fig.text(.05,.11,'The sole global N13 alias is {56,57}; both have only two N12 states and cannot fit these sixteen-positive samples.\nWitnesses are truth-aware existence certificates. Known successes and new failure diagnostics remain separate.',fontsize=11,linespacing=1.5)
    fig.text(.05,.045,f"BP AdamW: not measured. All 1,408 cases and all 1,541 models retained; no new source draws or oracle labels.\nStudy {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s. Full census and earlier ledger inherited by hash.",fontsize=10.6,color='#444',linespacing=1.5)
    for ext in ['png','svg']:fig.savefig(PUBLIC/f'identifiability-ceilings.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [path,Path(__file__)]},output_sha256={f'identifiability-ceilings.{ext}':hashlib.sha256((PUBLIC/f'identifiability-ceilings.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
