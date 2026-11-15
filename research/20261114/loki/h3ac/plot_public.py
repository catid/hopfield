"""Four finite-bank replication outcomes from tracked, audited exports only."""
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3ac'
def main():
    path=PUBLIC/'summary.json';s=json.loads(path.read_text());assert s['status']=='validated';banks=s['bank_results'];plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3ac-public'})
    fig,axes=plt.subplots(1,2,figsize=(14,8));fig.subplots_adjust(left=.07,right=.97,bottom=.35,top=.76,wspace=.28)
    fig.suptitle('Does the matched-time proposal gain survive independent source-bank fits?',x=.07,y=.96,ha='left',fontsize=16.5,fontweight='bold')
    fig.text(.07,.895,f"Four-bank conjunction: {s['decision'].upper()} · four independent 128-board draws · one finite rule family",fontsize=11.8,fontweight='bold')
    fig.text(.07,.84,'BP AdamW, original 4,000-update fit; bank-specific support and temperature; frozen kernels and source-selected proposal caps',fontsize=10.7)
    colors=['#c44e52','#19876b'];x=np.arange(4)
    for k in[0,1]:
        values=[b['rows'][k]['certified_success_pct']for b in banks];axes[0].bar(x+(k-.5)*.32,values,width=.3,color=colors[k],alpha=.85)
        for xi,v in zip(x+(k-.5)*.32,values):axes[0].annotate(f'{v:.1f}',(xi,v),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
    for k,b in enumerate(banks):
        ceiling=b['exact_zero_set_capacity']['certified_primary_ceiling_pct'];axes[0].plot([k-.34,k+.34],[ceiling,ceiling],color='#222',ls='--',lw=1.4)
    axes[0].set_ylim(0,110);axes[0].set_ylabel('Certified valid completion (%)');axes[0].set_xticks(x,[f'Bank {k}'for k in x]);axes[0].set_title('A   Fresh-target quality and exact capacity',loc='left',fontweight='bold',pad=12)
    axes[1].axhline(50,color='#c44e52',ls='--',lw=1.5)
    for k,b in enumerate(banks):
        p=b['primary'];gain=p['gain_pp'];axes[1].errorbar(k,gain,yerr=[[gain-p['one_sided_98_75_lower_pp']],[p['one_sided_98_75_upper_pp']-gain]],fmt='o',color='#19876b',capsize=5,lw=2);axes[1].annotate(f"+{gain:.2f} pp\ntime ratio {b['target_time_ratio']:.3f}",(k,gain),xytext=(0,15),textcoords='offset points',ha='center',fontsize=9)
    axes[1].set_ylim(45,85);axes[1].set_xlim(-.5,3.5);axes[1].set_xticks(x,[f'Bank {k}'for k in x]);axes[1].set_ylabel('Permutation minus arbitrary (percentage points)');axes[1].set_title('B   Required gain: 50 percentage points',loc='left',fontweight='bold',pad=12)
    for ax in axes:ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor=colors[0],label='BP AdamW, arbitrary proposals'),Patch(facecolor=colors[1],label='BP AdamW, permutation proposals'),Line2D([0],[0],color='#222',ls='--',label='Exact F-capacity ceiling')],loc='lower left',bbox_to_anchor=(.065,.26),ncol=3,frameon=False,fontsize=10)
    fig.text(.07,.215,'All methods share the same saved permutation starts. Source median times must match within 5%; fresh target medians within 10%.',fontsize=10.4,fontweight='bold')
    fig.text(.07,.155,'One common fresh 4,096-input cohort, three search seeds per input, 20 chains. All earlier Loki search and calibration keys are excluded.\nBounds are 98.75% one-sided per bank: familywise 95% lower endpoints, and separately 95% upper endpoints; not a joint two-sided 95% interval.',fontsize=10,linespacing=1.5)
    fig.text(.07,.085,f"Full cost: {s['total_gpu_worker_seconds']/60:.2f} GPU-worker minutes including fits, source calibration, warmups and targets.\nAll {s['audited_timed_target_states']:,} scored states independently audited. Bank draws overlap within the same 176-board source population.",fontsize=10,color='#444',linespacing=1.5)
    fig.text(.07,.028,'Search seeds repeat search; the four independently sampled banks provide distinct deterministic fitted energies. This is conditional finite-bank evidence.',fontsize=9.7,color='#444')
    for ext in['png','svg']:fig.savefig(PUBLIC/f'source-bank-replication.{ext}',dpi=180,facecolor='white',metadata={'Date':None}if ext=='svg'else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in[path,Path(__file__)]},output_sha256={f'source-bank-replication.{ext}':hashlib.sha256((PUBLIC/f'source-bank-replication.{ext}').read_bytes()).hexdigest()for ext in['png','svg']});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])
if __name__=='__main__':main()
