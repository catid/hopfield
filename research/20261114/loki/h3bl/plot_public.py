"""Separate posterior expected-cost endpoint from retrospective realized costs."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3bl'


def main():
    path=PUBLIC/'summary.json';s=json.loads(path.read_text());gpath=PUBLIC/'capacity-gate.json';g=json.loads(gpath.read_text());assert s['status']=='validated';plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3bl-public'})
    fig,(ax,emp)=plt.subplots(1,2,figsize=(13,9),gridspec_kw={'width_ratios':[1.05,1]});fig.subplots_adjust(left=.08,right=.95,bottom=.32,top=.71,wspace=.30)
    reduction=100*s['relative_expected_reduction']['display_float'];fig.suptitle(f'Exact tree planning reduces posterior expected query cost by {reduction:.2f}%',x=.04,y=.96,ha='left',fontsize=18,fontweight='bold');fig.text(.04,.90,'PRIMARY SUPPORTED: exact case-weighted improvement exceeds the frozen 10% threshold.',fontsize=12,fontweight='bold');fig.text(.04,.825,'125 examined checkpoint sets / 701 case mappings; every tree still identifies all remaining candidates within four queries.\nGuaranteed nonincrease is a correctness property. The 10% reduction is the separate empirical endpoint.',fontsize=11.1,linespacing=1.5)
    old=[r['old_expected_additional_queries']['display_float'] for r in g['groups']];new=[r['optimal_expected_additional_queries']['display_float'] for r in g['groups']];sizes=[12+3*r['case_multiplicity'] for r in g['groups']];ax.scatter(old,new,s=sizes,c=['#2d7d66' if r['strict_expected_improvement'] else '#6c8a9e' for r in g['groups']],alpha=.65,edgecolor='white',lw=.4);ax.plot([.85,4],[.85,4],ls='--',lw=1,color='#888');ax.set_xlim(.85,4.05);ax.set_ylim(.85,4.05);ax.set_aspect('equal');ax.set_xlabel('First-lex tree: expected additional queries');ax.set_ylabel('Optimal weighted tree: expected additional queries');ax.set_title('All 125 sets; marker area reflects case count',fontsize=11,pad=12);ax.grid(alpha=.12)
    labels=['First-lex\nH3bj trees','Weighted\noptimal trees','Unchanged\nMAP continuation'];values=[s['descriptive_old_continuation_cost'],s['descriptive_optimized_continuation_cost'],s['descriptive_baseline_MAP_continuation_cost']];emp.bar(range(3),values,color=['#517d9b','#4e9b7f','#a88a6d'],width=.62)
    for i,v in enumerate(values):emp.text(i,v+16,str(v),ha='center',fontsize=12)
    emp.set_xticks(range(3),labels);emp.set_ylim(0,1510);emp.set_ylabel('Continuation labels along examined true-model branches');emp.set_title('Retrospective realized costs: descriptive only',fontsize=11,pad=12);emp.grid(axis='y',alpha=.12);emp.set_axisbelow(True);emp.text(2,1090,'37 ties\nremain',ha='center',fontsize=10,color='white')
    fig.text(.04,.245,'Case-weighted posterior mean: 1.95725 → 1.68348 additional queries. 77 sets / 307 checkpoints improve strictly.',fontsize=11.8,fontweight='bold');fig.text(.04,.17,'Every canonical split and decisive child cost retained; exact integer weights and rational threshold comparison.\nThe selected source posterior defines this objective; no calibration, population-risk, or minimum worst-depth claim.',fontsize=11.1,linespacing=1.5);fig.text(.04,.095,'Representation-only repair of incomplete H3bk: Python’s decimal-output guard blocked the original final receipt.\nAll compiler inputs and proof bytes match that run. This is the original frozen endpoint, not independent confirmation.',fontsize=10.8,linespacing=1.5,color='#444');fig.text(.04,.035,f"BP AdamW: not measured. Zero new source draws or membership labels. {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s.",fontsize=10.7,color='#444')
    for ext in ['png','svg']:fig.savefig(PUBLIC/f'weighted-tree-cost.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [path,gpath,Path(__file__)]},output_sha256={f'weighted-tree-cost.{ext}':hashlib.sha256((PUBLIC/f'weighted-tree-cost.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
