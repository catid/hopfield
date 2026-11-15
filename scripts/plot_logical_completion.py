"""Independent-cohort evidence for the unchanged logical-completion target."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    first=ROOT/'results/research/foureyes/logical_completion/summary.json'
    second=ROOT/'results/research/foureyes/logical_completion_confirm/summary.json'
    a=json.loads(first.read_text())['comparisons'][0];b=json.loads(second.read_text())
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,4.7));fig.subplots_adjust(left=.27,right=.96,top=.75,bottom=.30)
    for y,row,color in [(1,a,'#498da4'),(0,b,'#ba6267')]:
        lo,hi=row['paired_input95_interval_pp'];gain=row['gain_pp']
        ax.errorbar(gain,y,xerr=[[gain-lo],[hi-gain]],fmt='o',color=color,capsize=5,markersize=8)
        ax.text(1.37,y,f"+{gain:.3f} pp\n{row['decision']}",va='center',fontsize=10,color=color)
    ax.axvline(0,color='#abb1b8',lw=1);ax.axvline(1,color='#a67136',ls='--',label='Frozen +1 pp target')
    ax.set_yticks([1,0],['Initial 10,000 inputs','Fresh 40,000 inputs']);ax.set_ylim(-.6,1.6);ax.set_xlim(-.05,1.78)
    ax.set_xlabel('Completed minus original accuracy (percentage points)');ax.legend(frameon=False,loc='upper left',fontsize=10)
    fig.suptitle('Adding logically implied edges helps, but the fresh gain is below one point',x=.04,ha='left',fontsize=14,fontweight='bold')
    fig.text(.04,.81,'Same 13 fixed BP AdamW graphs and unchanged one-point target; no outcome pooling.',fontsize=11)
    fig.text(.04,.07,'Zero sets are exactly unchanged; Sudoku geometry is supplied to complete the graphs.\nBoth studies have zero wrong stops. Intervals resample inputs, holding models and search seed fixed.\n40k confirmation: 77.6885% → 78.3006%; 720,000 saved boards independently audited.',fontsize=10,color='#52616e')
    out=ROOT/'assets/figures'
    for ext in ['png','svg']:fig.savefig(out/f'logical-completion-confirmation.{ext}',dpi=180)
    (out/'logical-completion-confirmation-provenance.json').write_text(json.dumps(dict(renderer=str(Path(__file__).relative_to(ROOT)),inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [first,second]}),indent=2)+'\n')


if __name__=='__main__':main()
