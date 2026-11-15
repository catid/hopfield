"""Deterministic H1cg recovery and scalar-mixture calibration diagnostics."""
from pathlib import Path
import hashlib,json,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source=ROOT/'results/research/ripper/cell_channel/summary.json';data=json.loads(source.read_text())
    plt.rcParams.update({'font.size':10,'svg.hashsalt':'h1cg-cell-channel-v1'});fig,axes=plt.subplots(1,2,figsize=(11.2,4.9))
    values=[g['success'] for g in data['grouped']];bars=axes[0].bar(['Supplied cell channels\n(privileged map)','Estimated scalar'],values,color=['#777777','#0072B2'],width=.6)
    for bar,value in zip(bars,values):axes[0].text(bar.get_x()+bar.get_width()/2,value+.5,str(value),ha='center')
    axes[0].axhline(60,linestyle='--',color='#333333',linewidth=1,label='Separate absolute targets: 60')
    axes[0].set_ylim(0,80);axes[0].set_ylabel('Exact graphs out of 72');axes[0].set_title('Same observations, different channel information');axes[0].legend(frameon=False,fontsize=8,loc='upper left')
    rates=[r['estimator']['rate_numerator'] for r in data['rows'][1::2] if r['estimator']['fit_allowed']];bins=sorted(set(rates));counts=[rates.count(k) for k in bins]
    if bins:
        bars=axes[1].bar(bins,counts,color='#009E73',width=.6)
        for bar,value in zip(bars,counts):axes[1].text(bar.get_x()+bar.get_width()/2,value+.3,str(value),ha='center')
    effective=100*(8/9)*(1-math.sqrt(1-(9/8)*(8393/51200)))
    axes[1].axvline(effective,color='#333333',linestyle='--',linewidth=1,label='Nominal mixture moment target')
    axes[1].set_xlabel('Estimated scalar noise rate (%)');axes[1].set_ylabel('Banks');axes[1].set_ylim(0,max(counts,default=0)+8);axes[1].set_title('27 cells each at 3.75%, 7.5% and 15%');axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    e=data['endpoints'];c=data['paired_changes'];fig.suptitle('H1cg: within-bank channel heterogeneity',fontsize=12)
    fig.text(.5,.06,f"Exact targets: supplied {e['supplied_exact']['decision']}, scalar {e['scalar_exact']['decision']}. Regression target: {e['paired_regressions']['decision']}.",ha='center',fontsize=9)
    fig.text(.5,.02,f"Scalar gains {len(c['gains'])}, losses {len(c['losses'])}; all 72 banks retained. Supplied control has privileged channel-map information.",ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.13,1,.93]);folder=ROOT/'assets/figures';png,svg=[folder/f'cell_channel.{ext}' for ext in ['png','svg']]
    fig.savefig(png,dpi=180,metadata={'Software':'Matplotlib'});fig.savefig(svg,metadata={'Date':None});plt.close(fig)
    provenance=dict(study='H1cg',input=str(source.relative_to(ROOT)),input_sha256=sha(source),script=str(Path(__file__).resolve().relative_to(ROOT)),script_sha256=sha(__file__),outputs={str(p.relative_to(ROOT)):sha(p) for p in [png,svg]},matplotlib=matplotlib.__version__,numpy=np.__version__)
    (folder/'cell_channel-provenance.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
