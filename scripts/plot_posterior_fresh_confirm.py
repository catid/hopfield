"""Deterministic fresh H1ca original, hard-decoded, and posterior arm comparison."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source=ROOT/'results/research/ripper/posterior_fresh_confirm/summary.json';data=json.loads(source.read_text())
    plt.rcParams.update({'font.size':10,'svg.hashsalt':'h1ca-posterior-fresh-confirm-v1'})
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.5))
    values=[g['success'] for g in data['grouped']]
    bars=axes[0].bar(['Single270','Hard triple90','Posterior triple90'],values,color=['#777777','#E69F00','#0072B2'],width=.65)
    for bar,value in zip(bars,values):axes[0].text(bar.get_x()+bar.get_width()/2,max(1,value-7),str(value),ha='center',color='white',fontweight='bold')
    axes[0].axhline(61,ls='--',color='#333333',lw=1,label='Posterior target: 61')
    axes[0].set_ylim(0,76);axes[0].set_ylabel('Exact graphs out of 64');axes[0].set_title('Fresh three-arm confirmation')
    axes[0].legend(frameon=False,fontsize=8,loc='upper left')
    labels=['vs single270','vs hard triple90'];x=np.arange(2)
    gains=[len(data['paired_changes'][key]['posterior_gains']) for key in ['versus_single270','versus_hard_triple90']]
    losses=[len(data['paired_changes'][key]['posterior_losses']) for key in ['versus_single270','versus_hard_triple90']]
    for delta,counts,color,label in [(-.18,gains,'#0072B2','Posterior gains'),(.18,losses,'#D55E00','Posterior losses')]:
        bars=axes[1].bar(x+delta,counts,width=.35,color=color,label=label)
        for bar,count in zip(bars,counts):axes[1].text(bar.get_x()+bar.get_width()/2,count+.15,str(count),ha='center')
    axes[1].set_xticks(x,labels);axes[1].set_ylabel('Paired banks');axes[1].set_ylim(0,max(6,max(gains+losses)+4))
    axes[1].legend(frameon=False,fontsize=8,loc='upper left')
    exact,loss=[data['endpoints'][k]['decision'] for k in ['posterior_exact','paired_loss']]
    axes[1].set_title(f'Posterior exact ≥61: {exact}\nLosses vs single ≤3: {loss}',fontsize=10)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('H1ca: fixed posterior scoring tested on 64 fresh paired banks',fontsize=12)
    unknown=sum(g['unresolved'] for g in data['grouped'])
    fig.text(.5,.05,f'All 192 arms retained; {unknown} unresolved. Original single and hard decoder controls rerun on the fresh cohort.',ha='center',fontsize=9)
    fig.text(.5,.015,'Each deployment uses 1,244,160 noisy queries; shared three-arm physical union is 2,073,600.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.105,1,.91]);folder=ROOT/'assets/figures'
    png,svg=[folder/f'posterior_fresh_confirm.{ext}' for ext in ['png','svg']]
    fig.savefig(png,dpi=180,metadata={'Software':'Matplotlib'});fig.savefig(svg,metadata={'Date':None});plt.close(fig)
    provenance=dict(study='H1ca',input=str(source.relative_to(ROOT)),input_sha256=sha(source),script=str(Path(__file__).resolve().relative_to(ROOT)),script_sha256=sha(__file__),
                    matplotlib=matplotlib.__version__,numpy=np.__version__,output_sha256={str(p.relative_to(ROOT)):sha(p) for p in [png,svg]})
    (folder/'posterior_fresh_confirm-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');print(json.dumps(provenance))


if __name__=='__main__':main()
