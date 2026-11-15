"""Fresh-bank magnitude-weight confirmation from public measurements."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/research/foureyes/canonical_confirm/summary.json'
OUT=ROOT/'assets/figures'


def main():
    s=json.loads(SOURCE.read_text());p=s['primary'];fig,axes=plt.subplots(1,3,figsize=(15,6.3))
    fig.suptitle('Coefficient weighting: fresh banks and fresh puzzles',x=.06,ha='left',weight='bold',fontsize=19)
    labels=['BP AdamW','PL uniform','PL magnitude','Counts']
    for i,r in enumerate(s['grouped']):
        axes[0].bar(i,r['correct_certified_pct'],color='#168777',label='Correct certified'if i==0 else None)
        axes[0].bar(i,r['correct_capped_pct'],bottom=r['correct_certified_pct'],color='#d49a3a',label='Correct capped'if i==0 else None)
        axes[0].text(i,r['accuracy_pct']+2,f"{r['accuracy_pct']:.2f}%",ha='center',fontsize=9.5,weight='bold')
    axes[0].set(ylim=(0,105),xticks=range(4),xticklabels=labels,ylabel='Canonical exact-match accuracy (%)');axes[0].tick_params(axis='x',labelsize=8.4);axes[0].legend(frameon=False,loc='upper left',fontsize=8.5)
    axes[0].set_title('A   Fixed recipes and controls',loc='left',weight='bold',fontsize=12)
    for ax,value,ci,threshold,title,xlabel in [
        (axes[1],p['gain_pp'],p['gain95_interval_pp'],50,'B   Gain target: ≥50 pp','Magnitude minus uniform (pp)'),
        (axes[2],p['accuracy_pct'],p['accuracy95_interval_pct'],60,'C   Accuracy target: ≥60%','Magnitude accuracy (%)')]:
        ax.axvline(threshold,color='#bf565e',ls='--');ax.errorbar(value,.5,xerr=[[value-ci[0]],[ci[1]-value]],fmt='o',color='#168777',capsize=5,lw=2)
        ax.text(.04,.82,f'{value:.2f}\n95% interval [{ci[0]:.2f}, {ci[1]:.2f}]',transform=ax.transAxes,fontsize=11)
        ax.set(xlim=(0,100),ylim=(0,1),yticks=[],xlabel=xlabel);ax.set_title(title,loc='left',weight='bold',fontsize=12)
    for ax in axes:ax.grid(alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    with np.load(SOURCE.parent/'audited-outcome-bits.npz',allow_pickle=False)as z:
        stops=dict(zip(map(str,z['jobs']),np.unpackbits(z['stopped'],axis=1,count=5000).sum(axis=1)))
    certificates=sum(int(stops[r['job']])for r in s['rows']if r['arm']=='magnitude')
    fig.text(.06,.21,f"Frozen conjunction: {p['decision'].upper()}. Magnitude: {certificates:,} certificates; {p['wrong_stops']} wrong stops / 320,000 evaluations.",fontsize=12,weight='bold')
    fig.text(.06,.13,'64 fresh N600 banks with nine corrupted free digits per board; 5,000 reserved official puzzles; one search seed.\nFixed λ=1/30000; median-positive-J/4 threshold; weights1 or1/16; four chains ×262,144 moves. No target tuning.',fontsize=10,linespacing=1.5)
    fig.text(.06,.05,'Two marginal paired two-way 95% intervals, with 5,000 draws each. BP AdamW and counts use separately fitted supports.\nThe official corpus is reused; this is a new-bank/new-input confirmation, not an external-distribution test.',fontsize=9.8,color='#555',linespacing=1.5)
    fig.subplots_adjust(left=.06,right=.98,top=.85,bottom=.34,wspace=.3)
    for ext in ['png','svg']:fig.savefig(OUT/f'canonical-confirm.{ext}',dpi=180,facecolor='white',metadata={'Date':None}if ext=='svg'else None)
    plt.close(fig);(OUT/'canonical-confirm-provenance.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),matplotlib=matplotlib.__version__,numpy=np.__version__),indent=2)+'\n')


if __name__=='__main__':main()
