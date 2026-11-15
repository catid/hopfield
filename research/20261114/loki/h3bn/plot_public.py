"""Keep the actual-call falsification separate from the expected-cost objective."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3bn'


def main():
    path=PUBLIC/'summary.json';s=json.loads(path.read_text());assert s['status']=='validated';expected_old=s['descriptive_case_weighted_expected_old_sum']['display_float']/1067;expected_new=s['descriptive_case_weighted_expected_optimal_sum']['display_float']/1067;actual_change=100*(s['new_continuation_calls']/s['old_continuation_calls']-1);expected_change=100*s['descriptive_expected_relative_reduction']['display_float'];plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3bn-public'})
    fig,(ax,actual)=plt.subplots(1,2,figsize=(13,8.5));fig.subplots_adjust(left=.09,right=.94,bottom=.32,top=.71,wspace=.34)
    fig.suptitle('Lower posterior expected cost produced more actual membership calls',x=.04,y=.96,ha='left',fontsize=18,fontweight='bold');fig.text(.04,.895,f'PRIMARY FALSIFIED: 2,913 actual continuation calls; required ≤2,588 for a 5% reduction.',fontsize=12.5,fontweight='bold');fig.text(.04,.82,'Same archived eight-positive banks and unchanged four-query checkpoints; 1,354/1,408 exact cases preserved.\nAll 439 feasible sets compiled before any new labels; the 54 infeasible cases and 287 prior successes are retained.',fontsize=11.1,linespacing=1.5)
    ax.bar([0,1],[expected_old,expected_new],color=['#557f9c','#59977d'],width=.62);ax.set_xticks([0,1],['First-lex tree','Weighted optimum']);ax.set_ylabel('Posterior expected additional queries');ax.set_ylim(0,max(expected_old,expected_new)*1.28);ax.set_title(f'Descriptive expected objective: −{expected_change:.2f}%',fontsize=12,pad=14)
    for i,v in enumerate([expected_old,expected_new]):ax.text(i,v+.055,f'{v:.4f}',ha='center',fontsize=12)
    actual.bar([0,1],[2725,2913],color=['#557f9c','#b66b54'],width=.62);actual.set_xticks([0,1],['Original paid calls','New paid calls']);actual.set_ylabel('Actual continuation membership calls');actual.set_ylim(0,3500);actual.set_title(f'Actual cost: +{actual_change:.2f}% (+188 calls)',fontsize=12,pad=14);actual.axhline(2588,ls='--',color='#777',lw=1);actual.text(.5,2475,'primary requires ≤2,588',ha='center',fontsize=10.5,color='#555',bbox=dict(facecolor='white',edgecolor='none',pad=1))
    for i,v in enumerate([2725,2913]):actual.text(i,v+65,str(v),ha='center',fontsize=12)
    for a in [ax,actual]:a.grid(axis='y',alpha=.13);a.set_axisbelow(True)
    fig.text(.04,.24,'The expected objective improved 17.16%; it did not predict a realized cost benefit on these selected truths.',fontsize=12,fontweight='bold');fig.text(.04,.165,'5,577 inherited MAP calls + 2,913 newly charged calls = 8,490 policy calls, versus 8,302 originally.\n2,020 new queries reacquire archived same-case answers; 893 query boards were absent from that parent case.',fontsize=11.1,linespacing=1.5);fig.text(.04,.085,'Archived-source intervention, not independent confirmation. No new failures is a correctness invariant.\nExact expected-cost optimization remains valid under its specified source posterior; population calibration is untested.',fontsize=10.9,color='#444',linespacing=1.5);fig.text(.04,.03,f"BP AdamW: not measured. Zero new source examples. Study {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s.",fontsize=10.7,color='#444')
    for ext in ['png','svg']:fig.savefig(PUBLIC/f'expected-versus-actual-cost.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [path,Path(__file__)]},output_sha256={f'expected-versus-actual-cost.{ext}':hashlib.sha256((PUBLIC/f'expected-versus-actual-cost.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
