"""Deterministic H1cc composite-likelihood comparison against archived controls."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source=ROOT/'results/research/ripper/pair_likelihood/summary.json';data=json.loads(source.read_text())
    plt.rcParams.update({'font.size':10,'svg.hashsalt':'h1cc-pair-likelihood-v1'})
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.5));x=np.arange(2)
    old=[g['old_exact'] for g in data['grouped']];new=[g['success'] for g in data['grouped']]
    for delta,values,color,label in [(-.18,old,'#777777','Archived H1ca control'),(.18,new,'#0072B2','Pair composite likelihood')]:
        bars=axes[0].bar(x+delta,values,width=.35,color=color,label=label)
        for bar,value in zip(bars,values):axes[0].text(bar.get_x()+bar.get_width()/2,max(1,value-7),str(value),ha='center',color='white',fontweight='bold')
    axes[0].set_xticks(x,['Single270\n(original control)','Triple90\n(posterior control)']);axes[0].set_ylim(0,80);axes[0].set_ylabel('Exact graphs out of 64')
    axes[0].set_title('Unchanged H1ca query observations');axes[0].legend(frameon=False,fontsize=8,loc='upper left')
    changes=data['paired_changes']['triple90_pair_likelihood'];counts=[len(changes['gains']),len(changes['losses'])]
    bars=axes[1].bar(['Triple90 gains','Triple90 regressions'],counts,color=['#0072B2','#D55E00'],width=.6)
    for bar,value in zip(bars,counts):axes[1].text(bar.get_x()+bar.get_width()/2,value+.1,str(value),ha='center')
    axes[1].set_ylim(0,max(5,max(counts)+2));axes[1].set_ylabel('Paired banks vs archived posterior')
    exact,reg=[data['endpoints'][k]['decision'] for k in ['triple_exact','posterior_regressions']]
    axes[1].set_title(f'Triple90 exact ≥63: {exact}\nZero posterior regressions: {reg}',fontsize=10)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('H1cc: nonlinear pair likelihood from repeated noisy labels',fontsize=12)
    unknown=sum(g['unresolved'] for g in data['grouped']);numeric=sum(g['numerical_unresolved'] for g in data['grouped'])
    fig.text(.5,.05,f'All 128 new arms retained; {unknown} unresolved ({numeric} numerical). Certified log intervals and exact LP proofs.',ha='center',fontsize=9)
    fig.text(.5,.015,'Examined-cohort objective comparison; no new observations and no full-graph likelihood claim.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.105,1,.91]);folder=ROOT/'assets/figures';png,svg=[folder/f'pair_likelihood.{ext}' for ext in ['png','svg']]
    fig.savefig(png,dpi=180,metadata={'Software':'Matplotlib'});fig.savefig(svg,metadata={'Date':None});plt.close(fig)
    provenance=dict(study='H1cc',input=str(source.relative_to(ROOT)),input_sha256=sha(source),script=str(Path(__file__).resolve().relative_to(ROOT)),script_sha256=sha(__file__),
                    matplotlib=matplotlib.__version__,numpy=np.__version__,output_sha256={str(p.relative_to(ROOT)):sha(p) for p in [png,svg]})
    (folder/'pair_likelihood-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');print(json.dumps(provenance))


if __name__=='__main__':main()
