"""Two fixed new-family outcomes, rendered only from tracked exports."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3ad'
def main():
    path=PUBLIC/'summary.json';s=json.loads(path.read_text());assert s['status']=='validated';families=s['family_results'];plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3ad-public'})
    fig,axes=plt.subplots(1,2,figsize=(13.6,8));fig.subplots_adjust(left=.07,right=.97,bottom=.36,top=.76,wspace=.28)
    fig.suptitle('Does the fixed matched-time recipe transfer to two new rule families?',x=.07,y=.96,ha='left',fontsize=17,fontweight='bold')
    fig.text(.07,.895,f"Two-family conjunction: {s['decision'].upper()} · complete N12 sources · independently verified complete N13 targets",fontsize=11.6,fontweight='bold')
    fig.text(.07,.84,'Same 4,000-update BP AdamW recipe, positive-F derivative, saved permutation starts and frozen search kernels',fontsize=10.8)
    colors=['#c44e52','#19876b'];labels=[]
    for i,f in enumerate(families):
        labels.append('Queen + zebra (2,3)'if i==0 else'Queen + (2,5) leaper')
        if not f['rows']:
            for ax in axes:ax.text(i,.5,f['decision'].replace('_','\n'),transform=ax.get_xaxis_transform(),ha='center',va='center',fontsize=9)
            continue
        for k,r in enumerate(f['rows']):
            x=i+(k-.5)*.32;v=r['certified_success_pct'];axes[0].bar(x,v,width=.3,color=colors[k],alpha=.85);axes[0].annotate(f'{v:.2f}',(x,v),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9.4)
        ceiling=f['exact_zero_set_capacity']['certified_primary_ceiling_pct'];axes[0].plot([i-.34,i+.34],[ceiling,ceiling],ls='--',color='#222',lw=1.4)
        p=f['primary'];gain=p['gain_pp'];axes[1].errorbar(i,gain,yerr=[[gain-p['one_sided_97_5_lower_pp']],[p['one_sided_97_5_upper_pp']-gain]],fmt='o',color='#19876b',capsize=5,lw=2);axes[1].annotate(f"+{gain:.2f} pp\nfresh time ratio {f['target_time_ratio']:.3f}",(i,gain),xytext=(0,15),textcoords='offset points',ha='center',fontsize=10)
    axes[0].set_ylim(0,112);axes[0].set_ylabel('Certified valid completion (%)');axes[0].set_title('A   Quality and exact F capacity',loc='left',fontweight='bold',pad=12);axes[1].axhline(50,color='#c44e52',ls='--');axes[1].set_ylim(0,120);axes[1].set_ylabel('Permutation minus arbitrary (percentage points)');axes[1].set_title('B   Required gain: 50 percentage points',loc='left',fontweight='bold',pad=12)
    for ax in axes:ax.set_xlim(-.5,1.5);ax.set_xticks([0,1],labels,fontsize=9.8);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor=colors[0],label='BP AdamW, arbitrary proposals'),Patch(facecolor=colors[1],label='BP AdamW, permutation proposals'),Line2D([0],[0],ls='--',color='#222',label='Exact F-capacity ceiling')],loc='lower left',bbox_to_anchor=(.065,.275),ncol=3,frameon=False,fontsize=9.8)
    counts='; '.join(f"{f['source_patterns']:,} source → {f['target_patterns']:,} target solutions"for f in families);fig.text(.07,.225,counts+'. Source counts differ; the fixed recipe uses every source pattern.',fontsize=10.3,fontweight='bold')
    fig.text(.07,.16,'Two separate fresh cohorts of 4,096 inputs; 20 chains and three search seeds per input. Source/fresh time gates: ±5% / ±10%.\n97.5% one-sided endpoints per family give adjusted lower and upper families separately; they are not a joint two-sided 95% interval.',fontsize=10.1,linespacing=1.5)
    fig.text(.07,.085,f"Full cost: {s['total_gpu_worker_seconds']/60:.2f} GPU-worker minutes, including fitting, source calibration, warmups and targets.\nComplete independent enumeration: {s['enumeration_gate']['aggregate_cpu_seconds']:.3f} CPU seconds. All {s['audited_timed_target_states']:,} scored states audited.",fontsize=10.1,color='#444',linespacing=1.5)
    fig.text(.07,.03,'Two prospectively fixed families from a known structural grid; this does not establish a result over a representative sample of all rule families.',fontsize=9.8,color='#444')
    for ext in['png','svg']:fig.savefig(PUBLIC/f'new-family-replication.{ext}',dpi=180,facecolor='white',metadata={'Date':None}if ext=='svg'else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in[path,Path(__file__)]},output_sha256={f'new-family-replication.{ext}':hashlib.sha256((PUBLIC/f'new-family-replication.{ext}').read_bytes()).hexdigest()for ext in['png','svg']});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])
if __name__=='__main__':main()
