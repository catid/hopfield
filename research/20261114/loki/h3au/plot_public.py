"""Plot the one-prior countermeasure, preserving the remaining failed bank."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3au'


def main():
    paths=[PUBLIC/'summary.json',PUBLIC/'posthoc-control-comparison.json'];s=json.loads(paths[0].read_text());comparison=json.loads(paths[1].read_text());assert s['status']=='validated'
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3au-public'})
    fig,axes=plt.subplots(1,2,figsize=(14,7.7));fig.subplots_adjust(left=.075,right=.96,bottom=.31,top=.75,wspace=.31)
    fig.suptitle('One declared complexity prior fixes seven banks, but leaves one failure',x=.065,y=.95,ha='left',fontsize=16.5,fontweight='bold')
    fig.text(.065,.88,'FALSIFIED: the all-bank requirement fails; seven MAP models are exact and one retains a false (7,10) orbit',fontsize=11.4,fontweight='bold')
    fig.text(.065,.818,'P(k) = 1/3 for k = 0, 1, 2; uniform subsets within each k. Same 1,541 candidates and sixteen positive labels.',fontsize=10.8)
    for index in range(8):
        current=next(r for r in s['rows'] if r['index']==index and r['primary']);old=next(r for r in comparison['imported_rows'] if r['index']==index and r['primary']);odds=float(comparison['banks'][index]['prior_ML_models'][0]['old_ML_to_true_MAP_weight_odds_decimal'])
        axes[0].bar(index-.17,old['capacity_pct'],.32,color='#a0afb4',label='Published H3at ML' if index==0 else None);axes[0].bar(index+.17,current['capacity_pct'],.32,color='#26786b',label='New hierarchical MAP' if index==0 else None);axes[0].text(index+.17,current['capacity_pct']-10,f"{current['capacity_pct']:.0f}" if current['capacity_pct']==100 else f"{current['capacity_pct']:.1f}",ha='center',color='white',fontsize=7.6)
        axes[1].bar(index,odds,.6,color='#ae5d43' if odds>1 else '#26786b');axes[1].text(index,odds+.035,f'{odds:.3f}',ha='center',fontsize=9)
    axes[0].axhline(95,color='#ad4841',ls='--',lw=1.2);axes[0].set_ylim(0,113);axes[0].set_ylabel('Exact valid-input capacity (%)');axes[0].set_title('A   Seven reach 100%; L2 remains at 89.66%',loc='left',fontsize=11.6,fontweight='bold',pad=14);axes[0].legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2,fontsize=9.4)
    axes[1].axhline(1,color='#ad4841',ls='--',lw=1.2);axes[1].set_ylim(0,1.33);axes[1].set_ylabel('Posterior weight ratio: old ML pair / true singleton');axes[1].set_title('B   The false pair still wins in L2 by 1.106×',loc='left',fontsize=11.6,fontweight='bold',pad=14)
    for ax in axes:ax.set_xticks(range(8),['Z0','Z1','Z2','Z3','L0','L1','L2','L3']);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.text(.065,.18,'All eight MAP maxima are unique and have zero invalid states. The failed graph retains 72 false edges.',fontsize=10.9,fontweight='bold')
    fig.text(.065,.115,'The prior gives a particular singleton 27× the prior mass of a particular pair; no coefficient was tuned.\nL2’s likelihood advantage is large enough to overcome that fixed penalty. All exact weights and the failed model are preserved.',fontsize=10.6,linespacing=1.5)
    fig.text(.065,.045,f"Adaptive countermeasure on the same seen banks and target universe; one chosen complexity prior, no new labels or BP fits.\nSource normalizers reused with full checks; target states freshly enumerated. Study: {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s.",fontsize=10.2,color='#444',linespacing=1.4)
    for ext in ('png','svg'):fig.savefig(PUBLIC/f'hierarchical-prior.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[Path(__file__)]},output_sha256={f'hierarchical-prior.{ext}':hashlib.sha256((PUBLIC/f'hierarchical-prior.{ext}').read_bytes()).hexdigest() for ext in ('png','svg')});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
