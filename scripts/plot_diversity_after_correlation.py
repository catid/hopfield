"""Deterministic H1ci acquisition-policy exact recovery and paired outcomes."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source=ROOT/'results/research/ripper/diversity_after_correlation/summary.json';data=json.loads(source.read_text())
    plt.rcParams.update({'font.size':10,'svg.hashsalt':'h1ci-diversity-after-correlation-v1'})
    fig,axes=plt.subplots(1,2,figsize=(11.2,4.9));groups=data['grouped'];values=[g['success'] for g in groups]
    bars=axes[0].bar(['90 boards × 3 responses\n(correlated repeats)','270 boards × 1 response\n(distinct within each bank)'],values,color=['#777777','#0072B2'],width=.6)
    for bar,value in zip(bars,values):axes[0].text(bar.get_x()+bar.get_width()/2,value+.5,str(value),ha='center')
    axes[0].axhline(60,linestyle='--',color='#333333',linewidth=1,label='Diverse-arm target: 60')
    axes[0].set_ylim(0,80);axes[0].set_ylabel('Exact graphs out of 72');axes[0].set_title('Same noisy-response budget');axes[0].legend(frameon=False,fontsize=8,loc='upper left')
    pairs=data['paired_changes'];names=['gains','losses','both_exact','both_resolved_failure','any_unresolved'];counts=[len(pairs[k]) for k in names]
    bars=axes[1].bar(range(5),counts,color=['#009E73','#D55E00','#0072B2','#777777','#CC79A7'],width=.65)
    for bar,value in zip(bars,counts):axes[1].text(bar.get_x()+bar.get_width()/2,value+.5,str(value),ha='center')
    axes[1].set_xticks(range(5),['Diverse\ngains','Diverse\nlosses','Both\nexact','Both\nfailed','Any\nunresolved'])
    axes[1].set_ylabel('Paired banks');axes[1].set_ylim(0,80);axes[1].set_title('All paired outcomes retained')
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    e=data['endpoints'];net=e['net_recovery_gain'];fig.suptitle('H1ci: source diversity after correlated-repeat failure',fontsize=12)
    fig.text(.5,.065,f"Diverse exact target: {e['diverse_exact']['decision']}. Net recovery gain [{net['lower']}, {net['upper']}] vs 48: {net['decision']}.",ha='center',fontsize=9)
    fig.text(.5,.02,'19,440 noisy responses per deployment and bank; larger source requirement explicit. First 90 rows of responses and masks shared.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.14,1,.93]);folder=ROOT/'assets/figures';png,svg=[folder/f'diversity_after_correlation.{ext}' for ext in ['png','svg']]
    fig.savefig(png,dpi=180,metadata={'Software':'Matplotlib'});fig.savefig(svg,metadata={'Date':None});plt.close(fig)
    provenance=dict(study='H1ci',input=str(source.relative_to(ROOT)),input_sha256=sha(source),script=str(Path(__file__).resolve().relative_to(ROOT)),script_sha256=sha(__file__),outputs={str(p.relative_to(ROOT)):sha(p) for p in [png,svg]},matplotlib=matplotlib.__version__,numpy=np.__version__)
    (folder/'diversity_after_correlation-provenance.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
