"""Deterministic H1ce recovery and estimated-channel distribution."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source=ROOT/'results/research/ripper/estimated_channel/summary.json';data=json.loads(source.read_text());g=data['grouped'][0]
    plt.rcParams.update({'font.size':10,'svg.hashsalt':'h1ce-estimated-channel-v1'})
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.7));values=[g['old_exact'],g['success']]
    bars=axes[0].bar(['Supplied nominal 3/40\n(archived)','Estimated channel'],values,color=['#777777','#0072B2'],width=.6)
    for bar,value in zip(bars,values):axes[0].text(bar.get_x()+bar.get_width()/2,max(1,value-6),str(value),ha='center',color='white',fontweight='bold')
    axes[0].axhline(60,color='#333333',linestyle='--',linewidth=1,label='Estimated-arm threshold: 60')
    axes[0].set_ylim(0,72);axes[0].set_ylabel('Exact graphs out of 64');axes[0].set_title('All 64 examined banks retained');axes[0].legend(frameon=False,fontsize=8,loc='upper left')
    rates=[row['estimator']['rate_numerator'] for row in data['rows'] if row['estimator']['fit_allowed']]
    bins=sorted(set(rates));counts=[rates.count(k) for k in bins]
    if bins:
        bars=axes[1].bar([str(k) for k in bins],counts,color='#009E73',width=.65)
        for bar,value in zip(bars,counts):axes[1].text(bar.get_x()+bar.get_width()/2,value+.3,str(value),ha='center')
    axes[1].set_xlabel('Estimated noise rate (%)');axes[1].set_ylabel('Banks');axes[1].set_ylim(0,max(5,max(counts,default=0)+6))
    axes[1].set_title('Exact disagreement estimate, fixed 1% grid')
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    endpoints=data['endpoints']['estimated_channel'];change=data['paired_changes']['estimated_channel']
    fig.suptitle('H1ce: infer the channel from repeated observed labels',fontsize=12)
    fig.text(.5,.055,f"Exact target: {endpoints['exact']['decision']}; regression target: {endpoints['regressions']['decision']}. Gains {len(change['gains'])}, regressions {len(change['losses'])}.",ha='center',fontsize=9)
    fig.text(.5,.015,f"{g['unresolved']} unresolved, including {g['estimator_unresolved']} estimates. Same observations estimate the channel and score pairs; no independence claim.",ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.12,1,.93]);folder=ROOT/'assets/figures';png,svg=[folder/f'estimated_channel.{ext}' for ext in ['png','svg']]
    fig.savefig(png,dpi=180,metadata={'Software':'Matplotlib'});fig.savefig(svg,metadata={'Date':None});plt.close(fig)
    provenance=dict(study='H1ce',input=str(source.relative_to(ROOT)),input_sha256=sha(source),script=str(Path(__file__).resolve().relative_to(ROOT)),script_sha256=sha(__file__),outputs={str(p.relative_to(ROOT)):sha(p) for p in [png,svg]},matplotlib=matplotlib.__version__,numpy=np.__version__)
    (folder/'estimated_channel-provenance.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
