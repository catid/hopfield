"""Exact hypothetical acquisition reliability, preserving current versus future distinction."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3av'


def main():
    paths=[PUBLIC/'summary.json',ROOT/'results/research/loki/h3au/summary.json'];s=json.loads(paths[0].read_text());old=json.loads(paths[1].read_text());assert s['status']==old['status']=='validated'
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3av-public'})
    fig,ax=plt.subplots(figsize=(13,7.8));fig.subplots_adjust(left=.09,right=.96,bottom=.34,top=.75)
    fig.suptitle('Sixteen more hypothetical positives do not ensure reliable recovery',x=.065,y=.95,ha='left',fontsize=16.6,fontweight='bold')
    fig.text(.065,.88,'FALSIFIED: only 4 / 8 initial banks reach ≥95% probability of safe, high-capacity updated MAP models',fontsize=11.7,fontweight='bold')
    fig.text(.065,.818,'Exact conditional acquisition: choose 16 unseen positives from the true population minus each fixed initial 16',fontsize=10.9)
    for r in s['rows']:
        index=r['index'];before=next(x for x in old['rows'] if x['index']==index and x['primary']);color='#b66b42' if before['decision']=='supported' and r['success_pct']<95 else '#28796f';ax.bar(index,r['success_pct'],.64,color=color);ax.text(index,r['success_pct']-8,f"{r['success_pct']:.2f}%",ha='center',fontsize=10,color='white')
    ax.axhline(95,color='#a9433e',ls='--',lw=1.3);ax.set_ylim(0,115);ax.set_ylabel('P(all updated MAP ties safe and ≥95% capacity), %');ax.set_xticks(range(8),['Z0','Z1','Z2','Z3','L0','L1','L2','L3']);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.text(.065,.24,'Four probabilities are exactly 100% by integer equality. The others remain below the frozen 95% requirement.',fontsize=11,fontweight='bold')
    fig.text(.065,.17,'Orange: three initially correct models can become wrong after hypothetical extra positives. If a false orbit remains\nunwitnessed, the stronger 32-label likelihood can overcome the unchanged complexity prior. L2 improves to 91.69%.',fontsize=10.7,linespacing=1.5)
    fig.text(.065,.1,'No new observed labels are consumed. This enumerates the future acquisition distribution using evaluator-only source populations.\nAll 104 masks (46 positive-mass), every tie and 14 complete supports are retained, including invalid supports with zero probability.',fontsize=10.4,linespacing=1.5)
    fig.text(.065,.035,f"Unchanged prior: equal mass for k = 0, 1, 2, then uniform subsets; same seen initial banks and target universe.\nStudy: {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s. Two exact joint-count methods agree on every mask.",fontsize=10.2,color='#444',linespacing=1.4)
    for ext in ('png','svg'):fig.savefig(PUBLIC/f'conditional-acquisition.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[Path(__file__)]},output_sha256={f'conditional-acquisition.{ext}':hashlib.sha256((PUBLIC/f'conditional-acquisition.{ext}').read_bytes()).hexdigest() for ext in ('png','svg')});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
