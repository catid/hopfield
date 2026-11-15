"""Deterministic H1cd robustness comparison with archived nominal control."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source=ROOT/'results/research/ripper/channel_misspecification/summary.json';data=json.loads(source.read_text())
    plt.rcParams.update({'font.size':10,'svg.hashsalt':'h1cd-channel-misspecification-v1'})
    arms=[g['arm'] for g in data['grouped']];groups=data['grouped'];assert all(g['old_exact']==62 for g in groups)
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.7));x=np.arange(3)
    values=[groups[0]['success'],62,groups[1]['success']]
    bars=axes[0].bar(x,values,color=['#0072B2','#777777','#009E73'],width=.65)
    for bar,value in zip(bars,values):axes[0].text(bar.get_x()+bar.get_width()/2,value-6,str(value),ha='center',color='white',fontweight='bold')
    axes[0].axhline(60,color='#333333',linestyle='--',linewidth=1,label='New-arm threshold: 60')
    axes[0].set_xticks(x,['Assumed 3/80','Nominal 3/40\n(archived)','Assumed 3/20']);axes[0].set_ylim(0,72)
    axes[0].set_ylabel('Exact graphs out of 64');axes[0].set_title('True acquisition noise: 3/40');axes[0].legend(frameon=False,fontsize=8,loc='upper left')
    x=np.arange(2)
    for delta,key,color,label in [(-.18,'gains','#0072B2','Gains'),(.18,'losses','#D55E00','Regressions')]:
        vals=[len(data['paired_changes'][arm][key]) for arm in arms]
        bars=axes[1].bar(x+delta,vals,width=.35,color=color,label=label)
        for bar,value in zip(bars,vals):axes[1].text(bar.get_x()+bar.get_width()/2,value+.1,str(value),ha='center')
    axes[1].axhline(2,color='#333333',linestyle='--',linewidth=1,label='Regression maximum: 2')
    axes[1].set_xticks(x,['Assumed 3/80','Assumed 3/20']);axes[1].set_ylim(0,max(5,max(len(data['paired_changes'][a][k]) for a in arms for k in ['gains','losses'])+2))
    axes[1].set_ylabel('Paired banks vs nominal control');axes[1].set_title('Both assumptions prespecified');axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('H1cd: likelihood robustness to a wrong noise assumption',fontsize=12)
    decisions=[' / '.join(data['endpoints'][a][k]['decision'] for k in ['exact','regressions']) for a in arms]
    fig.text(.5,.055,f'Exact / regression targets: 3/80 {decisions[0]}; 3/20 {decisions[1]}.',ha='center',fontsize=9)
    unknown=sum(g['unresolved'] for g in groups)
    fig.text(.5,.015,f'All 128 new arms retained; {unknown} unresolved. Examined-cohort comparison; no new observations.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.12,1,.93]);folder=ROOT/'assets/figures';png,svg=[folder/f'channel_misspecification.{ext}' for ext in ['png','svg']]
    fig.savefig(png,dpi=180,metadata={'Software':'Matplotlib'});fig.savefig(svg,metadata={'Date':None});plt.close(fig)
    provenance=dict(study='H1cd',input=str(source.relative_to(ROOT)),input_sha256=sha(source),script=str(Path(__file__).resolve().relative_to(ROOT)),script_sha256=sha(__file__),outputs={str(f.relative_to(ROOT)):sha(f) for f in [png,svg]},matplotlib=matplotlib.__version__,numpy=np.__version__)
    (folder/'channel_misspecification-provenance.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
