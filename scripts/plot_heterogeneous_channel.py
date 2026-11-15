"""Deterministic H1cf per-channel recovery and estimated noise calibration."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source=ROOT/'results/research/ripper/heterogeneous_channel/summary.json';data=json.loads(source.read_text())
    plt.rcParams.update({'font.size':10,'svg.hashsalt':'h1cf-heterogeneous-channel-v1'})
    fig,axes=plt.subplots(1,2,figsize=(11.2,4.9));colors=['#777777','#0072B2'];positions=np.arange(3)
    for ai,(group,color,label) in enumerate(zip(data['grouped'],colors,['Supplied nominal','Estimated channel'])):
        values=[s['success'] for s in group['strata']];bars=axes[0].bar(positions+(ai-.5)*.34,values,width=.32,color=color,label=label)
        for bar,value in zip(bars,values):axes[0].text(bar.get_x()+bar.get_width()/2,value+.25,str(value),ha='center',fontsize=9)
    axes[0].set_xticks(positions,['3.75%','7.5%','15%']);axes[0].set_xlabel('Nominal acquisition noise');axes[0].set_ylabel('Exact graphs out of 24');axes[0].set_ylim(0,28);axes[0].legend(frameon=False,fontsize=9,loc='upper right')
    axes[0].set_title('Fresh paired methods, same observed labels')
    for ri,(nominal,color) in enumerate(zip([3.75,7.5,15],['#009E73','#0072B2','#D55E00'])):
        rates=[r['estimator']['rate_numerator'] for r in data['rows'][1::2] if r['seed']%3==ri and r['estimator']['fit_allowed']];bins=sorted(set(rates));counts=[rates.count(k) for k in bins]
        axes[1].scatter(bins,counts,s=65,color=color,label=f'Nominal {nominal:g}%');axes[1].axvline(nominal,color=color,alpha=.55,linestyle='--',linewidth=1)
    axes[1].set_xlabel('Estimated noise rate (%)');axes[1].set_ylabel('Banks at each estimated rate');axes[1].set_ylim(0,26);axes[1].set_title('Fixed 1% estimator grid');axes[1].legend(frameon=False,fontsize=9)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    endpoints=data['endpoints'];changes=data['paired_changes'];fig.suptitle('H1cf: fresh heterogeneous-channel confirmation',fontsize=12)
    fig.text(.5,.06,f"Exact targets (at least 60/72): supplied {endpoints['supplied_exact']['decision']}, estimated {endpoints['estimated_exact']['decision']}.",ha='center',fontsize=9)
    fig.text(.5,.02,f"Regression target (at most 2): {endpoints['paired_regressions']['decision']}. Paired gains {len(changes['gains'])}, losses {len(changes['losses'])}; all 72 banks retained.",ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.13,1,.93]);folder=ROOT/'assets/figures';png,svg=[folder/f'heterogeneous_channel.{ext}' for ext in ['png','svg']]
    fig.savefig(png,dpi=180,metadata={'Software':'Matplotlib'});fig.savefig(svg,metadata={'Date':None});plt.close(fig)
    provenance=dict(study='H1cf',input=str(source.relative_to(ROOT)),input_sha256=sha(source),script=str(Path(__file__).resolve().relative_to(ROOT)),script_sha256=sha(__file__),outputs={str(p.relative_to(ROOT)):sha(p) for p in [png,svg]},matplotlib=matplotlib.__version__,numpy=np.__version__)
    (folder/'heterogeneous_channel-provenance.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
