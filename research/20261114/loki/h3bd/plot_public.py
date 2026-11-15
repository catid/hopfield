"""Plot exact paired bank outcomes for all fixed truth families."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3bd'


def main():
    paths=[PUBLIC/'summary.json',ROOT/'research/20261114/loki/h3bd/protocol.json'];s=json.loads(paths[0].read_text());p=json.loads(paths[1].read_text());assert s['status']=='validated';groups={}
    for r in s['results']:groups.setdefault((r['archive'],r['family']),[]).append(r)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3bd-public'})
    fig,ax=plt.subplots(figsize=(14,13));fig.subplots_adjust(left=.31,right=.93,bottom=.23,top=.77)
    fig.suptitle('MAP-guided questions resolve 79 additional source banks',x=.05,y=.96,ha='left',fontsize=18,fontweight='bold')
    fig.text(.05,.91,'SUPPORTED: 106/108 exact ≥ 81 required, with all 8 singleton cases exact. All-108 secondary: FALSIFIED.',fontsize=12,fontweight='bold')
    fig.text(.05,.85,'Same 108 banks, same 16 source positives, same 1,541-model vocabulary and four-query cap.\nPaired outcomes: 27 both exact · 79 guided only · 0 balanced only · 2 both unresolved.',fontsize=11.5,linespacing=1.5)
    labels=[]
    for row,((archive,family),cases) in enumerate(groups.items()):
        cfg=p['cases'][cases[0]['case']];labels.append(('Singleton ' if archive=='H3ao' else f'Pair f{family}: ')+' + '.join(f'({a},{b})' for a,b in cfg['true_orbits']));a=sum(r['balanced_exact'] for r in cases);b=sum(r['exact_true_singleton'] for r in cases)
        ax.plot([a,b],[row,row],c='#ccc',lw=2,zorder=1);ax.scatter(a,row-.10,c='#527b9c',s=45,marker='o',label='Balanced counts: 27/108' if row==0 else None,zorder=3);ax.scatter(b,row+.10,c='#26796b',s=55,marker='D',label='MAP-guided: 106/108' if row==0 else None,zorder=3)
        if b<4:ax.text(b+.12,row+.15,f'{b}/4 guided',va='center',fontsize=10,color='#a24e45')
        if row%2==0:ax.axhspan(row-.48,row+.48,color='#777',alpha=.035)
    ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_xlim(-.15,4.5);ax.set_xticks([0,1,2,3,4]);ax.set_xlabel('Exact unique models among four frozen source banks');ax.grid(axis='x',alpha=.14);ax.set_axisbelow(True);ax.legend(loc='upper center',bbox_to_anchor=(.5,1.075),ncol=2,frameon=False)
    fig.text(.05,.16,'Guided: 278 binary labels. Balanced: 422. All earlier successes preserved; two family-7 banks still tie.',fontsize=11.5,fontweight='bold')
    fig.text(.05,.10,'The source MAP prior and likelihood were unchanged; questions maximize alternatives disagreeing with the guide.\nKnown candidate solver predictions + noiseless binary oracle; reused and duplicate banks are not independent trials.',fontsize=11,linespacing=1.5)
    fig.text(.05,.04,f"BP AdamW: not measured. Full literal census inherited by hash from H3bc; fresh complete filter/score parity.\nStudy: {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s. No same-oracle comparison to positive-only samples.",fontsize=10.8,color='#444',linespacing=1.5)
    for ext in ['png','svg']:fig.savefig(PUBLIC/f'guided-queries.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths+[Path(__file__)]},output_sha256={f'guided-queries.{ext}':hashlib.sha256((PUBLIC/f'guided-queries.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
