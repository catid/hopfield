"""Source calibration, realized fresh costs and quality from tracked evidence."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3ab'
def main():
    path=PUBLIC/'summary.json';s=json.loads(path.read_text());assert s['status']=='validated';cal=s['calibration_selection'];groups=cal['all_calibration_groups'];ref=groups['arbitrary-c20000']['median_seconds'];perms=sorted([g for g in groups.values()if g['kernel']=='permutation'],key=lambda g:g['cap']);q=s['primary'];ratio=s['target_timing']['median_permutation_to_arbitrary_ratio']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3ab-public'})
    fig,axes=plt.subplots(1,3,figsize=(15,7.7));fig.subplots_adjust(left=.055,right=.98,bottom=.39,top=.75,wspace=.32)
    status=s['decision'].replace('_',' ');fig.suptitle('Does the proposal-quality advantage survive a measured-compute comparison?',x=.055,y=.96,ha='left',fontsize=18,fontweight='bold')
    fig.text(.055,.89,f"Matched-compute decision: {status}. Quality-only gain: {q['accuracy_difference_pp']:+.2f} pp ({s['quality_decision']}).",fontsize=11.7,fontweight='bold')
    fig.text(.055,.83,f"Source-selected caps frozen at 20,000 arbitrary / {cal['selected_permutation_cap']:,} permutation · identical weights, shared starts, temperature endpoints",fontsize=10.9)
    axes[0].axhspan(ref*.95,ref*1.05,color='#c44e52',alpha=.12);axes[0].axhline(ref,color='#c44e52',ls='--',lw=1.3)
    axes[0].plot([g['cap']/1000 for g in perms],[g['median_seconds']for g in perms],'-o',color='#19876b')
    for g in perms:
        for r in g['trials']:axes[0].scatter(g['cap']/1000+(r['gpu']-.5)*.08,r['worker_solve_seconds'],s=12,color='#19876b',alpha=.5)
    selected=next(g for g in perms if g['cap']==cal['selected_permutation_cap']);axes[0].scatter(selected['cap']/1000,selected['median_seconds'],marker='*',s=170,color='#8c5eb5',zorder=5)
    axes[0].set_xlabel('Permutation proposal cap (thousands)');axes[0].set_ylabel('Source-derived solve time (seconds)');axes[0].set_title('A   Source-only count selection',loc='left',fontweight='bold',pad=12)
    axes[0].text(.04,.95,f"Selected ratio {cal['source_median_time_ratio']:.3f}\nrequired 0.95–1.05",transform=axes[0].transAxes,va='top',fontsize=9.5)
    colors=['#c44e52','#19876b'];names=['BP AdamW\nArbitrary proposals','BP AdamW\nPermutation proposals'];freshref=s['rows'][0]['median_worker_solve_seconds']
    axes[1].axhspan(freshref*.9,freshref*1.1,color='#777',alpha=.13)
    for k,row in enumerate(s['rows']):
        v=row['median_worker_solve_seconds'];axes[1].bar(k,v,width=.5,color=colors[k],alpha=.75);axes[1].errorbar(k,v,yerr=[[v-row['minimum_worker_solve_seconds']],[row['maximum_worker_solve_seconds']-v]],color='#333',capsize=4,fmt='none');axes[1].annotate(f'{v:.3f}s',(k,v),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9.3)
        for shift,key,alpha,hatch in[(-.16,'certified_success_pct',.9,None),(.16,'certificate_first_accuracy_pct',.35,'//')]:
            value=row[key];axes[2].bar(k+shift,value,width=.29,color=colors[k],alpha=alpha,hatch=hatch);axes[2].annotate(f'{value:.2f}',(k+shift,value),xytext=(0,-14)if value>60 else(0,5),textcoords='offset points',ha='center',fontsize=8.5,bbox=dict(facecolor='white',edgecolor='none',pad=.1))
    for ax in axes[1:]:ax.set_xticks([0,1],names,fontsize=9)
    axes[1].set_ylim(0,max(r['maximum_worker_solve_seconds']for r in s['rows'])*1.28);axes[1].set_ylabel('Fresh-target median solve time (seconds)');axes[1].set_title('B   Realized cost, no retuning',loc='left',fontweight='bold',pad=12)
    axes[2].set_ylim(0,113);axes[2].set_ylabel('Valid completion accuracy (%)');axes[2].set_title('C   Quality at the frozen caps',loc='left',fontweight='bold',pad=12)
    for ax in axes:ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor='#777',alpha=.9,label='Certified primary'),Patch(facecolor='#777',alpha=.35,hatch='//',label='Certificate-first with energy fallback')],loc='lower left',bbox_to_anchor=(.05,.28),ncol=2,frameon=False,fontsize=10)
    fig.text(.055,.23,f"Fresh median time ratio = {ratio:.3f}; required 0.90–1.10. Timing gate {'passes'if s['target_timing']['matching_gate_passed']else'fails'}; the primary decision keeps this separate from quality.",fontsize=11,fontweight='bold')
    fig.text(.055,.16,'Calibration uses source-derived N13-shaped clues, three search streams on both GPUs, a fixed grid and at most one interpolation.\nAll calibration keys are excluded from the 4,096 fresh target inputs. Proposal counts are never adjusted after target costs are observed.',fontsize=10.1,linespacing=1.5)
    fig.text(.055,.085,f"Total GPU-worker time, including calibration, warmups and target searches: {s['total_gpu_worker_seconds']:.1f}s. Calibration costs are not omitted.\nThe endpoints of the source temperature schedule are fixed; changing caps also changes annealing duration. All 491,520 scored states are audited.",fontsize=9.8,color='#444',linespacing=1.5)
    fig.text(.055,.025,'One inherited BP AdamW fit; three search seeds clustered per input. Equal-compute interpretation requires the declared realized-time gate.',fontsize=9.5,color='#444')
    for ext in['png','svg']:fig.savefig(PUBLIC/f'matched-compute.{ext}',dpi=180,facecolor='white',metadata={'Date':None}if ext=='svg'else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in[path,Path(__file__)]},output_sha256={f'matched-compute.{ext}':hashlib.sha256((PUBLIC/f'matched-compute.{ext}').read_bytes()).hexdigest()for ext in['png','svg']})
    (PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])
if __name__=='__main__':main()
