"""Deterministic joint-endpoint figure; cost improvement cannot hide a lost case."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];P=ROOT/'results/research/loki/h3bz'


def main():
    src=P/'summary.json';s=json.loads(src.read_text());H,U=s['H_calls'],s['U_calls'];saving=100*(H-U)/H;t=s['exact_transitions']
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'h3bz-joint-no-loss'})
    fig,ax=plt.subplots(figsize=(12.8,9.5));fig.subplots_adjust(left=.12,right=.95,bottom=.39,top=.73)
    fig.suptitle(f"Joint hypothesis {s['decision'].upper()}: {t['H_only_exact']} lost identification",x=.04,y=.96,ha='left',fontsize=20,fontweight='bold')
    fig.text(.04,.895,f"{saving:.2f}% fewer calls and {s['U_exact']:,} versus {s['H_exact']:,} exact cases; per-case no-loss condition fails.",fontsize=13)
    fig.text(.04,.82,'All 1,024 fresh uniform-truth pairs completed. Both deeper policies execute and pay separately.\n'
             'U matches the truth prior; H keeps the original hierarchical fitted weights.',fontsize=11.5,linespacing=1.5)
    maps=[s['MAP_calls'][a] for a in ['H','U']];trees=[s['tree_calls'][a] for a in ['H','U']]
    ax.bar([0,1],maps,width=.5,color='#557f9c',label='MAP calls');ax.bar([0,1],trees,bottom=maps,width=.5,color='#bb805f',label='Tree calls')
    ax.set_xticks([0,1],['H: hierarchical fitted weights\nfixed deeper policy','U: uniform fitted weights\nsame deeper policy'])
    ax.set_ylabel('Total actual membership calls');ax.set_ylim(0,max(H,U)*1.25);ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True)
    ax.axhline(s['joint_primary']['passing_maximum'],color='#744848',ls='--',lw=1.2)
    ax.text(.5,s['joint_primary']['passing_maximum']*.93,f"Cost target ≤{s['joint_primary']['passing_maximum']:,}",ha='center',fontsize=10.5,
            color='#744848',bbox=dict(facecolor='white',edgecolor='none',pad=2))
    ax.legend(loc='upper left',frameon=False,ncol=2,bbox_to_anchor=(0,1.04))
    for i,total in enumerate([H,U]):ax.text(i,total+max(H,U)*.018,f'{total:,}',ha='center',fontsize=13,fontweight='bold')
    fig.text(.04,.30,f"Both exact: {t['both_exact']:,}      H only: {t['H_only_exact']}      U only: {t['U_only_exact']}      Neither: {t['neither_exact']}",fontsize=13,fontweight='bold')
    fig.text(.04,.235,'The prespecified joint endpoint requires ≥2% fewer calls AND no H-exact case lost by U.\n'
             'Higher aggregate recovery does not replace the per-case condition.',fontsize=11.5,linespacing=1.5)
    fig.text(.04,.16,'Case 549: U stops after four MAP calls with 19 candidates; H reaches a tree and identifies truth.\n'
             'All five discordant failures hit the >16-candidate routing limit; no global resource cap was reached.',fontsize=11.2,linespacing=1.5)
    fig.text(.04,.085,'Finite-cohort result; no population superiority claim. Full proofs retained outside Git.\n'
             'Diagnosis uses existing sealed transcripts and supplied predictions; no new oracle queries.',fontsize=10.7,color='#444',linespacing=1.5)
    fig.text(.04,.03,f"BP AdamW: not measured. {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s.",fontsize=10.5,color='#444')
    for ext in ['png','svg']:fig.savefig(P/f'joint-cost-and-exact.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [src,Path(__file__)]},
                 output_sha256={f'joint-cost-and-exact.{ext}':hashlib.sha256((P/f'joint-cost-and-exact.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']})
    (P/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
