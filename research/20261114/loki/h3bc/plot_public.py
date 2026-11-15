"""Compare initial ambiguity and exact four-query success by every truth family."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3bc'


def main():
    paths=[PUBLIC/'summary.json',ROOT/'research/20261114/loki/h3bc/protocol.json'];s=json.loads(paths[0].read_text());p=json.loads(paths[1].read_text());assert s['status']=='validated';groups={}
    for r in s['results']:groups.setdefault((r['archive'],r['family']),[]).append(r)
    plt.rcParams.update({'font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3bc-public'})
    fig=plt.figure(figsize=(15,13));ax=fig.add_axes([.31,.23,.32,.56]);bx=fig.add_axes([.71,.23,.21,.56],sharey=ax)
    fig.suptitle('Removing the complete-source class breaks four-query identification',x=.05,y=.96,ha='left',fontsize=18,fontweight='bold')
    fig.text(.05,.91,'FALSIFIED: 27/108 exact unique models — 8/8 singleton banks and 19/100 two-orbit banks.',fontsize=12.5,fontweight='bold')
    fig.text(.05,.85,'Every case begins with all 1,541 models filtered only by sixteen observed N12 positives.\nKnown candidate predictions and a noiseless binary N13 membership oracle remain supplied.',fontsize=11.3,linespacing=1.5)
    labels=[]
    for row,((archive,family),cases) in enumerate(groups.items()):
        cfg=p['cases'][cases[0]['case']];labels.append(('Singleton ' if archive=='H3ao' else f'Pair f{family}: ')+' + '.join(f'({a},{b})' for a,b in cfg['true_orbits']));counts=[len(r['initial_candidates']) for r in cases];success=sum(r['exact_true_singleton'] for r in cases);color='#26796b' if archive=='H3ao' else '#557d9a'
        ax.plot([min(counts),max(counts)],[row,row],color=color,lw=2)
        for j,count in enumerate(counts):ax.scatter(count,row+(j-1.5)*.10,c=color,s=20,zorder=3)
        bx.barh(row,success,color=color,height=.68);bx.text(success+.09,row,f'{success}/4',va='center',fontsize=9.6,color=color)
        if row%2==0:
            ax.axhspan(row-.48,row+.48,color='#777',alpha=.035);bx.axhspan(row-.48,row+.48,color='#777',alpha=.035)
    ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_xscale('log',base=2);ax.set_xlim(3,300);ax.set_xticks([4,8,16,32,64,128,256],[4,8,16,32,64,128,256]);ax.set_xlabel('Initial consistent candidates (log scale)');ax.set_title('Four frozen source banks per family',fontsize=11.5,fontweight='bold',pad=14);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    bx.tick_params(axis='y',left=False,labelleft=False);bx.set_xlim(0,4.8);bx.set_xticks([0,1,2,3,4]);bx.set_xlabel('Exact unique cases after ≤4 labels');bx.set_title('Every remaining tie fails',fontsize=11.5,fontweight='bold',pad=14);bx.grid(axis='x',alpha=.15);bx.set_axisbelow(True)
    fig.text(.05,.16,'81 cases remain ambiguous (2–14 models); initial sets span 4–232 models. All 422 query/answer receipts retained.',fontsize=11.5,fontweight='bold')
    fig.text(.05,.10,'Three M12=16 families repeat the same unordered source bank four times; the 108 cases are not independent.\nAll 20 ineligible truth-grid families remain public. Label cost: 16 source positives + 2–4 binary queries per case.',fontsize=11,linespacing=1.5)
    fig.text(.05,.04,f"BP AdamW: not measured. Earlier MAP accuracy uses a different endpoint; no random-control result is transferred.\nStudy: {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s. Candidate enumeration dominates runtime.",fontsize=10.8,color='#444',linespacing=1.5)
    for ext in ['png','svg']:fig.savefig(PUBLIC/f'full-class-queries.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths+[Path(__file__)]},output_sha256={f'full-class-queries.{ext}':hashlib.sha256((PUBLIC/f'full-class-queries.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
