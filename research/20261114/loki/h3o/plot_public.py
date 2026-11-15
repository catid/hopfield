"""Upward-transfer figure from tracked summary only."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[4];DEST=ROOT/'results/research/loki/h3o'
def main():
    source=DEST/'summary.json';s=json.loads(source.read_text());assert s['status']=='validated'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3o-public'})
    arms=[None,'n14-full','true-graph-oracle'];colors=['#777777','#19876b','#8c5eb5'];labels=['BP AdamW\nnot run','All 5,180 N14\nsource patterns','Supplied N15\ntarget oracle']
    fig,axes=plt.subplots(1,2,figsize=(12,7));fig.subplots_adjust(left=.075,right=.98,bottom=.40,top=.75,wspace=.23)
    fig.suptitle('Source-global zeros become false exclusions on a larger board',x=.075,y=.96,ha='left',fontsize=16.5,fontweight='bold')
    p=s['primary'];fig.text(.075,.89,f"Frozen N14-to-N15 noninferiority: {s['decision']}. Accuracy difference {p['accuracy_difference_pp']:+.3f} pp; margin −0.1 pp.",fontsize=11,fontweight='bold')
    fig.text(.075,.83,'4,096 fresh N15 superqueen inputs · 3 search seeds · all four source-global extra offsets retained',fontsize=10.6)
    for k,arm in enumerate(arms):
        if arm is None:
            for ax in axes:ax.text(k,45,'Not measured\non superqueens',ha='center',va='center',color=colors[k],fontsize=9)
            continue
        row=next(r for r in s['rows'] if r['arm']==arm)
        for offset,key,alpha,hatch in [(-.17,'original_accuracy_pct',.35,'//'),(.17,'certificate_first_accuracy_pct',.9,None)]:
            value=row[key];axes[0].bar(k+offset,value,width=.31,color=colors[k],alpha=alpha,hatch=hatch)
            axes[0].annotate(f'{value:.2f}',(k+offset,value),xytext=(0,6),textcoords='offset points',ha='center',fontsize=8)
        value=row['coverage_pct'];axes[1].bar(k,value,width=.62,color=colors[k],alpha=.87)
        axes[1].annotate(f'{value:.2f}%',(k,value),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
    for ax,title in zip(axes,['A   Accuracy from 20 chains (N20)','B   Distinct-superqueen coverage']):
        ax.set_xticks(range(3),labels,fontsize=9);ax.set_xlim(-.55,2.55);ax.set_ylim(0,114)
        ax.set_ylabel('Percent');ax.set_title(title,loc='left',fontweight='bold',pad=12);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor='#777',alpha=.35,hatch='//',label='Original lowest-energy selection'),Patch(facecolor='#777',alpha=.9,label='Certificate-first selection')],loc='lower left',bbox_to_anchor=(.07,.30),ncol=2,frameon=False,fontsize=9)
    fig.text(.075,.27,'Four source-global zero offsets add 24 false target pair exclusions; no true attack is missed.',fontsize=10.4,fontweight='bold')
    cap=next(v for v in s['posthoc_valid_certificate_capacity'] if v['arm']=='n14-full')
    fig.text(.075,.21,f"The learned energy admits a valid certificate for {cap['inputs_with_any_valid_certificate']:,} of 4,096 inputs and\n{cap['valid_input_completion_certificates']:,} of {cap['all_compatible_valid_completions']:,} compatible completions. All stopped boards satisfy the actual task.",fontsize=10,linespacing=1.5)
    fig.text(.075,.14,'The source counts and line rule are unchanged from the downward-transfer study. Target cardinality becomes 15,\nwith symbolic delta = 14 and tau = −105; temperature unit, margin, chain count and move cap remain inherited.',fontsize=9,color='#444',linespacing=1.5)
    fig.text(.075,.075,'All 32,516 N15 superqueen solutions were independently enumerated. Six distinct searches ran; no outcome mapping.\nSource-global absences are not transferable local rules, even when every stopped certificate remains correct.',fontsize=9,color='#444',linespacing=1.5)
    fig.text(.075,.025,'Original and certificate-first metrics are retained separately. This fixed-source boundary test does not estimate performance across source draws.',fontsize=8.8,color='#444')
    for ext in ['png','svg']:fig.savefig(DEST/f'upward-transfer.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,Path(__file__)]},output_sha256={f'upward-transfer.{ext}':hashlib.sha256((DEST/f'upward-transfer.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']})
    (DEST/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt['output_sha256']))
if __name__=='__main__':main()
