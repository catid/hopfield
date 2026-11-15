"""Deterministic H1cb exact truth-gap diagnostic figure."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source=ROOT/'results/research/ripper/posterior_residual_gap/summary.json';data=json.loads(source.read_text())
    plt.rcParams.update({'font.size':10,'svg.hashsalt':'h1cb-posterior-residual-gap-v1'})
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.5));x=np.arange(3)
    strict=[g['strict'] for g in data['grouped']];cooptimal=[g['cooptimal_nonexact'] for g in data['grouped']]
    for delta,values,color,label in [(-.18,strict,'#D55E00','Truth strictly worse'),(.18,cooptimal,'#0072B2','Truth cooptimal')]:
        bars=axes[0].bar(x+delta,values,width=.35,color=color,label=label)
        for bar,value in zip(bars,values):axes[0].text(bar.get_x()+bar.get_width()/2,value+.15,str(value),ha='center')
    axes[0].set_xticks(x,['Single270','Hard triple90','Posterior triple90']);axes[0].set_ylabel('Nonexact selected graphs')
    axes[0].set_ylim(0,max(5,max(strict+cooptimal)+4));axes[0].legend(frameon=False,fontsize=8,loc='upper left');axes[0].set_title('All 192 selected proofs rechecked')
    failed=[r for r in data['rows'] if r['arm']=='posterior_triple90' and not r['exact']];values=[r['gap_twice'] for r in failed]
    bars=axes[1].bar([f'Bank {r["seed"]}' for r in failed],values,color=['#D55E00' if v else '#0072B2' for v in values],width=.6)
    axes[1].set_yscale('symlog',linthresh=1);axes[1].set_ylim(0,max(2,max(values)*2));axes[1].set_ylabel('Twice objective gap (symlog scale)')
    for bar,value in zip(bars,values):axes[1].text(bar.get_x()+bar.get_width()/2,value*1.1+.05,str(value),ha='center')
    axes[1].set_title(f'At least 2 of 3 failures strictly exclude truth:\n{data["decision"]}',fontsize=10)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('H1cb: do residual posterior failures reflect a strict objective gap?',fontsize=12)
    fig.text(.5,.05,'Positive gap excludes truth from this optimum; zero gap establishes cooptimality, not uniqueness.',ha='center',fontsize=9)
    fig.text(.5,.015,'Reused H1ca observations and selected graphs; no new LPs, query records, noise, or source IDs.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.105,1,.91]);folder=ROOT/'assets/figures';png,svg=[folder/f'posterior_residual_gap.{ext}' for ext in ['png','svg']]
    fig.savefig(png,dpi=180,metadata={'Software':'Matplotlib'});fig.savefig(svg,metadata={'Date':None});plt.close(fig)
    provenance=dict(study='H1cb',input=str(source.relative_to(ROOT)),input_sha256=sha(source),script=str(Path(__file__).resolve().relative_to(ROOT)),script_sha256=sha(__file__),
                    matplotlib=matplotlib.__version__,numpy=np.__version__,output_sha256={str(p.relative_to(ROOT)):sha(p) for p in [png,svg]})
    (folder/'posterior_residual_gap-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');print(json.dumps(provenance))


if __name__=='__main__':main()
