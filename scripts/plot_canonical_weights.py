"""Public-only coefficient-weighting result, with certificate decomposition."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/research/foureyes/canonical_weights/summary.json'
OUT=ROOT/'assets/figures'


def main():
    s=json.loads(SOURCE.read_text());fig,axes=plt.subplots(1,2,figsize=(12.8,6.4))
    fig.suptitle('Can learned coefficient magnitudes repair unit-weight inference?',x=.07,ha='left',fontsize=18,weight='bold')
    for i,r in enumerate(s['grouped']):
        axes[0].bar(i,r['correct_certified_pct'],color='#168777',label='Correct certified'if i==0 else None)
        axes[0].bar(i,r['correct_capped_pct'],bottom=r['correct_certified_pct'],color='#d49a3a',label='Correct capped return'if i==0 else None)
        axes[0].text(i,r['accuracy_pct']+2,f"{r['accuracy_pct']:.2f}%",ha='center',weight='bold')
    axes[0].set(ylim=(0,105),ylabel='Canonical exact-match accuracy (%)',xticks=range(3),xticklabels=['Uniform','Magnitude','Shuffled'])
    axes[0].set_title('A   Identical positive support; changed weights',loc='left',fontsize=12,weight='bold');axes[0].legend(frameon=False,loc='upper left',fontsize=9)
    for i,e in enumerate(s['endpoints']):
        lo,hi=e['paired_two_way95_interval_pp'];g=e['gain_pp'];axes[1].errorbar(i,g,yerr=[[g-lo],[hi-g]],fmt='o',capsize=6,color='#168777',lw=2)
        axes[1].annotate(f"{g:+.2f} pp\n{e['decision']}",(i,hi),xytext=(0,12),textcoords='offset points',ha='center',fontsize=10)
    axes[1].axhline(10,color='#bf565e',ls='--');axes[1].set(xlim=(-.5,1.5),ylim=(-5,105),ylabel='Magnitude minus control (percentage points)',xticks=range(2),xticklabels=['Versus uniform','Versus shuffled'])
    axes[1].set_title('B   Required gains: ≥10 percentage points',loc='left',fontsize=12,weight='bold')
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    fig.text(.07,.20,f"Frozen conjunction: {s['decision'].upper()}. All weights stay positive on the inherited support; all true Sudoku edges remain present.",fontsize=11,weight='bold')
    fig.text(.07,.12,'Adaptive post-hoc mechanism study: 64 frozen weak-penalty PL fits; the same already-seen 10,000 puzzles; one search seed.\nMagnitude threshold = median(positive J)/4; strong 1, weak 1/16. Shuffling preserves the weight multiset. No geometry enters assignment.',fontsize=9.8,linespacing=1.5)
    fig.text(.07,.045,'Four chains × 262,144 moves, fixed temperatures and focus. Two marginal paired two-way 95% intervals; no simultaneous or fresh-generalization claim.\nBP AdamW is a separate inherited control in H4w; this intervention isolates inference weights on the weak-penalty PL support.',fontsize=9.4,color='#555',linespacing=1.5)
    fig.subplots_adjust(left=.07,right=.98,top=.86,bottom=.33,wspace=.27)
    for ext in ['png','svg']:fig.savefig(OUT/f'canonical-weights.{ext}',dpi=180,facecolor='white',metadata={'Date':None}if ext=='svg'else None)
    plt.close(fig)
    (OUT/'canonical-weights-provenance.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),matplotlib=matplotlib.__version__,numpy=np.__version__),indent=2)+'\n')


if __name__=='__main__':main()
