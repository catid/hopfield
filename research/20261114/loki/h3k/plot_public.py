"""Source-bank uncertainty and prespecified behavior, rendered from public data."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[4];DEST=ROOT/'results/research/loki/h3k'


def main():
    source=DEST/'summary.json';s=json.loads(source.read_text());assert s['status']=='validated'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3k-public'})
    fig,axes=plt.subplots(1,2,figsize=(13,6.8));fig.subplots_adjust(left=.07,right=.98,bottom=.34,top=.75,wspace=.24)
    fig.suptitle('Source-bank robustness under a fixed geometric prior',x=.07,y=.96,ha='left',fontsize=19,fontweight='bold')
    fig.text(.07,.89,'Frozen primary: 8 examples recover the exact four-line rule in 512 / 512 independent source draws.',fontsize=11,fontweight='bold')
    fig.text(.07,.83,'Exact one-sided 95% lower recovery bound: 99.4166% · fixed family, thresholds and target settings',fontsize=11)
    ax=axes[0];r=s['source_recovery'];x=np.array([v['source_patterns'] for v in r]);y=100*np.array([v['recovery_fraction'] for v in r])
    lower=100*np.array([v['exact95_lower'] for v in r]);upper=100*np.array([v['exact95_upper'] for v in r])
    ax.errorbar(x,y,yerr=[y-lower,upper-y],marker='o',color='#2878b5',capsize=4,linewidth=2)
    ax.scatter([8],[100],s=110,facecolors='none',edgecolors='#19876b',linewidths=2,zorder=5)
    ax.axhline(99,color='#777',linestyle='--',linewidth=1);ax.text(6.4,94.2,'99% primary threshold',fontsize=8.5,color='#666')
    for xx,yy,v in zip(x,y,r):ax.annotate(f"{v['exact_recovery_successes']}/512",(xx,yy),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
    ax.set_xscale('log',base=2);ax.set_xticks(x,[str(v) for v in x]);ax.set_ylim(64,108);ax.set_xlim(.85,19)
    ax.set_xlabel('N12 examples per source bank');ax.set_ylabel('Exact four-line recovery (%)');ax.grid(axis='y',alpha=.18)
    ax.set_title('A   Primary: 512 independent source draws',loc='left',fontweight='bold',pad=12)
    ax=axes[1]
    for draw in range(6):
        rows=[next(v for v in s['rows'] if v['draw']==draw and v['source_patterns']==n) for n in [1,8]]
        offset=(draw-2.5)*.018;color='#ca8a04' if draw==1 else '#8c5eb5' if draw==5 else '#19876b'
        label='Draw 1: spurious one-example line' if draw==1 else 'Draw 5: spurious one-example line' if draw==5 else 'Other four draws: exact rule' if draw==0 else None
        ax.plot(np.array([0,1])+offset,[v['coverage_pct'] for v in rows],marker='o',color=color,label=label,alpha=.9)
    ax.set_xticks([0,1],['1 example','8 examples']);ax.set_xlim(-.18,1.18);ax.set_ylim(0,100)
    ax.set_ylabel('N13 distinct-completion coverage (%)');ax.set_title('B   Behavior: first six prespecified draws',loc='left',fontweight='bold',pad=12)
    ax.grid(axis='y',alpha=.18);ax.legend(loc='lower right',frameon=False,fontsize=8)
    fig.text(.07,.245,'One-example recovery is 72.27% across all 512 draws. Failures retain extra lines; no fitted model misses a true line.',fontsize=10.4,fontweight='bold')
    fig.text(.07,.18,'The two prespecified failed one-example fits still reach 92.53% / 91.94% N20, but only 20.47% / 20.62% coverage.\nAll exact-rule models reach 100% N20 and 84.21% coverage; no wrong stopped chains occur.',fontsize=10,linespacing=1.5)
    fig.text(.07,.105,'Source intervals are exact one-sided 95% lower/upper endpoints over independent bank draws. Target results use\n2,048 fresh inputs and three search seeds; the six-draw behavior subset is descriptive, not the 512-draw primary.',fontsize=9,color='#444',linespacing=1.5)
    fig.text(.07,.035,'Coordinate geometry, the finite line family and q=13 are supplied priors. BP AdamW was not run.\nTwelve logical target models reduce to three exact energies: nine unique searches plus 27 explicit identity mappings.',fontsize=9,color='#444',linespacing=1.4)
    for ext in ['png','svg']:fig.savefig(DEST/f'source-bank-robustness.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,Path(__file__)]},
        output_sha256={f'source-bank-robustness.{ext}':hashlib.sha256((DEST/f'source-bank-robustness.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']})
    (DEST/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt['output_sha256']))


if __name__=='__main__':main()
