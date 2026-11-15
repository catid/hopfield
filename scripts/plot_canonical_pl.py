"""Render H4w from compact public measurements and exact proof summaries."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'results/research/foureyes/canonical_study'
OUT = ROOT/'assets/figures'


def main():
    paths = [PUBLIC/'summary.json', PUBLIC/'certificates.json']
    s,c = [json.loads(p.read_text()) for p in paths]
    labels = ['BP AdamW\n32 updates','Canonical PL\nλ = 1/6000','Canonical PL\nλ = 1/30000','Restricted\ncounts']
    colors = ['#68758d','#bf565e','#d49a3a','#168777']
    fig,axes = plt.subplots(1,3,figsize=(15,6.6))
    fig.suptitle('Numerical convergence does not guarantee correct constraint graphs',x=.06,ha='left',weight='bold',fontsize=18)
    for i,(r,color) in enumerate(zip(s['grouped'],colors)):
        axes[0].bar(i,r['accuracy_pct'],color=color)
        axes[0].text(i,r['accuracy_pct']+2,f"{r['accuracy_pct']:.2f}%",ha='center',fontsize=10)
        axes[1].bar(i,r['wrong_stops'],color=color)
        axes[1].text(i,r['wrong_stops']+20,str(r['wrong_stops']),ha='center',fontsize=10)
        sound = c['grouped'][i]['sound'];retained = r['all_true_retained']
        axes[2].bar(i-.16,retained,width=.3,color=color,alpha=.55)
        axes[2].bar(i+.16,sound,width=.3,color=color)
        axes[2].text(i-.16,retained+1,str(retained),ha='center',fontsize=9)
        axes[2].text(i+.16,sound+1,str(sound),ha='center',fontsize=9)
    titles = ['A   Correct returned answers','B   Incorrect certified stops','C   All true edges / proved sound']
    for ax,title in zip(axes,titles):
        ax.set_title(title,loc='left',weight='bold',fontsize=12,pad=12)
        ax.set_xticks(range(4),labels,fontsize=8.7);ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
    axes[0].set_ylim(0,100);axes[0].set_ylabel('Accuracy (%)')
    axes[1].set_ylim(0,1000);axes[1].set_ylabel('Wrong stops / 640,000 logical evaluations')
    axes[2].set_ylim(0,72);axes[2].set_ylabel('Models / 64')
    lo,hi = s['primary']['paired_two_way95_interval_pp']
    fig.text(.06,.20,f"Primary minus counts: {s['primary']['gain_pp']:+.3f} pp; paired two-way 95% interval [{lo:+.3f}, {hi:+.3f}]. Accuracy margin passes; zero-wrong-stop conjunction FAILS.",fontsize=11,weight='bold')
    fig.text(.06,.125,'64 fresh noisy N600 banks; 10,000 shared reserved puzzles; one search seed; four chains × 262,144 moves.\nCanonical fits use unrestricted J, exact digit invariance and 9λ L1; inference keeps J > 0 at unit weight. The weak-λ arm is descriptive.',fontsize=10,linespacing=1.5)
    fig.text(.06,.045,'Soundness uses post-hoc Sudoku geometry and exact count identities. Missing edges can be redundant; one unsound primary model has an explicit witness.\nBP AdamW has different regularization and finite augmentation. This comparison does not isolate optimizer choice or reproduce the paper’s training trajectory.',fontsize=9.5,color='#555',linespacing=1.5)
    fig.subplots_adjust(left=.06,right=.98,top=.85,bottom=.34,wspace=.32)
    for ext in ['png','svg']:
        fig.savefig(OUT/f'canonical-pl.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    plt.close(fig)
    receipt = dict(inputs_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),matplotlib=matplotlib.__version__,numpy=np.__version__)
    (OUT/'canonical-pl-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':
    main()
