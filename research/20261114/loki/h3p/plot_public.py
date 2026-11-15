"""Public fixed-model confirmation figure, using tracked initial and confirmation exports."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];DEST=ROOT/'results/research/loki/h3p'
def main():
    source=DEST/'summary.json';prior=DEST.parent/'h3o/summary.json';s=json.loads(source.read_text());old=json.loads(prior.read_text());assert s['status']==old['status']=='validated'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3p-public'})
    fig,axes=plt.subplots(1,3,figsize=(15,7));fig.subplots_adjust(left=.055,right=.98,bottom=.35,top=.75,wspace=.35)
    fig.suptitle('Prospective confirmation keeps the source and solver fixed',x=.055,y=.96,ha='left',fontsize=19,fontweight='bold')
    p=s['primary'];fig.text(.055,.89,f"16,384 fresh inputs: {s['decision']}. Learned-minus-oracle accuracy = {p['accuracy_difference_pp']:+.3f} pp; NI margin −0.1 pp.",fontsize=12,fontweight='bold')
    fig.text(.055,.83,'N14-to-N15 upward transfer · exact prior model files · 3 search seeds · initial 4,096 inputs excluded',fontsize=11)
    colors=['#777','#19876b','#8c5eb5'];arms=[None,'n14-full','true-graph-oracle'];labels=['BP AdamW\nnot run','Full N14\nhybrid','Supplied\noracle']
    for k,arm in enumerate(arms):
        if arm is None:
            for ax in [axes[0],axes[2]]:ax.text(k,45,'Not measured',ha='center',va='center',color=colors[k],fontsize=8)
            continue
        row=next(r for r in s['rows'] if r['arm']==arm)
        for ax,key in [(axes[0],'certificate_first_accuracy_pct'),(axes[2],'coverage_pct')]:
            v=row[key];ax.bar(k,v,width=.62,color=colors[k],alpha=.87);ax.annotate(f'{v:.3f}%',(k,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8.5)
    for ax,title in [(axes[0],'A   Accuracy from 20 chains'),(axes[2],'C   Distinct-solution coverage')]:
        ax.set_xticks(range(3),labels,fontsize=9);ax.set_ylim(0,112);ax.set_xlim(-.55,2.55);ax.set_ylabel('Percent');ax.set_title(title,loc='left',fontweight='bold',pad=12);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    ax=axes[1]
    for y,study,color in [(1,old,'#777'),(0,s,'#19876b')]:
        p=study['primary'];v=p['accuracy_difference_pp'];lo=p['accuracy_one_sided_95_lower_pp'];hi=p['accuracy_one_sided_95_upper_pp']
        ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o',capsize=5,color=color)
        ax.annotate(study['decision'],(v,y),xytext=(0,13),textcoords='offset points',ha='center',fontsize=9,color=color)
    ax.axvline(-.1,color='#b44',ls='--',label='−0.1 pp NI margin');ax.axvline(0,color='#bbb',lw=.8)
    ax.set_yticks([1,0],['Initial\n4,096 inputs','Confirmation\n16,384 inputs'],fontsize=9);ax.set_ylim(-.6,1.6)
    ax.set_xlabel('Learned minus oracle (percentage points)');ax.set_title('B   Separate paired comparisons',loc='left',fontweight='bold',pad=12);ax.grid(axis='x',alpha=.18)
    ax.legend(loc='lower left',fontsize=8,frameon=False)
    cap=next(x for x in s['posthoc_valid_certificate_capacity'] if x['arm']=='n14-full')
    fig.text(.055,.27,'The initial study remains inconclusive. The confirmation is a separately frozen, disjoint-cohort test.',fontsize=11,fontweight='bold')
    fig.text(.055,.205,f"Both models, temperature unit, margin and move cap are unchanged. The hybrid retains all 24 false target exclusions.\nIt admits a valid certificate for {cap['inputs_with_any_valid_certificate']:,} of 16,384 inputs and {cap['valid_input_completion_certificates']:,} of {cap['all_compatible_valid_completions']:,} compatible completions.",fontsize=10,linespacing=1.5)
    fig.text(.055,.135,'Intervals show the two one-sided 95% bootstrap endpoints (90% two-sided coverage); all three seeds cluster within each input.\nAccuracy noninferiority is the primary decision; coverage and certificate-capacity diagnostics are descriptive.',fontsize=9.5,color='#444',linespacing=1.5)
    fig.text(.055,.07,'The larger cohort was chosen after the initial inconclusive result, before any confirmation outcome was inspected.\nThis fixed-source, distribution-specific confirmation does not establish source-bank robustness or universal transfer.',fontsize=9.5,color='#444',linespacing=1.5)
    fig.text(.055,.025,'Original and certificate-first metrics remain separate in the tracked tables. No new optimizer fit, model pruning, source data or target-label tuning.',fontsize=9,color='#444')
    for ext in ['png','svg']:fig.savefig(DEST/f'upward-confirmation.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,prior,Path(__file__)]},output_sha256={f'upward-confirmation.{ext}':hashlib.sha256((DEST/f'upward-confirmation.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']})
    (DEST/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt['output_sha256']))
if __name__=='__main__':main()
