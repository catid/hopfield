"""Separate the continuation primary from the much smaller full-policy saving."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3br'


def main():
    path=PUBLIC/'summary.json';s=json.loads(path.read_text());assert s['status']=='validated';order=['firstlex','uniform','posterior'];labels=['First-lex','Uniform','Posterior'];values=[s['continuation_calls'][k] for k in order];totals=[s['policy_calls'][k] for k in order];saving=100*(1-values[2]/values[1]);total_saving=100*(1-totals[2]/totals[1]);limit=s['primary']['right']//100
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3br-public'})
    fig,(ax,total)=plt.subplots(1,2,figsize=(13,8.5));fig.subplots_adjust(left=.085,right=.95,bottom=.32,top=.72,wspace=.32)
    fig.suptitle('Matched prior: posterior trees save continuation calls',x=.04,y=.96,ha='left',fontsize=18,fontweight='bold')
    fig.text(.04,.89,f'PRIMARY SUPPORTED: {values[2]:,} posterior calls versus {values[1]:,} uniform; {saving:.2f}% reduction, required ≥2%.',fontsize=12.5,fontweight='bold')
    fig.text(.04,.82,'4,096 IID truths from the declared prior conditioned on at least eight source positives; 909 sampled models.\nAll policies identify the same 4,083 cases. All 13 infeasible cases, 3,855 prior successes and duplicates retained.',fontsize=11.2,linespacing=1.5)
    colors=['#9aa3aa','#557f9c','#60957b'];ax.bar(range(3),values,color=colors,width=.57);ax.set_xticks(range(3),labels);ax.set_ylabel('Actual continuation calls');ax.set_ylim(0,630);ax.set_title('Frozen continuation endpoint',fontsize=12.5,pad=14);ax.axhline(limit,ls='--',color='#6b5454',lw=1.2);ax.text(1.15,limit-47,f'Passing target ≤{limit:,}',ha='center',fontsize=10.5,color='#5b4545',bbox=dict(facecolor='white',edgecolor='none',pad=2))
    for i,v in enumerate(values):ax.text(i,v+14,f'{v:,}',ha='center',fontweight='bold')
    total.bar(range(3),totals,color=colors,width=.57);total.set_xticks(range(3),labels);total.set_ylabel('Calls including shared MAP stage');total.set_ylim(0,12500);total.set_title(f'Descriptive full-policy saving: {total_saving:.2f}%',fontsize=12.5,pad=14)
    for i,v in enumerate(totals):total.text(i,v+250,f'{v:,}',ha='center',fontweight='bold')
    for a in [ax,total]:a.grid(axis='y',alpha=.13);a.set_axisbelow(True)
    fig.text(.04,.24,'The 66-call saving is 13.52% of continuation cost, but only 0.63% after adding 9,933 shared MAP calls.',fontsize=11.8,fontweight='bold')
    fig.text(.04,.17,'Exact conditional-prior integer weights: 1,485 / 27 / 1 for zero / one / two added leaper orbits.\n1,522 eligible models and 4,436 equiprobable tickets; all 19 ineligible catalog rows retained as context.',fontsize=11.3,linespacing=1.5)
    fig.text(.04,.085,'Finite-cohort result under the declared prior; no population 2% guarantee or arbitrary-rule claim.\nPosterior expected nonincrease and equal per-case accuracy are correctness properties, separate from the primary.',fontsize=10.8,color='#444',linespacing=1.5)
    fig.text(.04,.03,f"BP AdamW: not measured. 11,344 charged membership calls. Study {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s.",fontsize=10.7,color='#444')
    for ext in ['png','svg']:fig.savefig(PUBLIC/f'matched-prior-cost.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [path,Path(__file__) ]},output_sha256={f'matched-prior-cost.{ext}':hashlib.sha256((PUBLIC/f'matched-prior-cost.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
