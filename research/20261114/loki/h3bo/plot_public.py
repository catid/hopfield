"""Show selected-sample log-score mismatch and descriptive fixed-bin reliability."""
import sys
sys.set_int_max_str_digits(0)
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3bo'
def number(x):return float(Fraction(int(x['numerator']),int(x['denominator'])))


def main():
    path=PUBLIC/'summary.json';bpath=PUBLIC/'calibration-bins.json';s=json.loads(path.read_text());bins=json.loads(bpath.read_text());assert s['status']=='validated';plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3bo-public'})
    fig,(loss,reliability)=plt.subplots(1,2,figsize=(13,8.5));fig.subplots_adjust(left=.08,right=.94,bottom=.31,top=.70,wspace=.32)
    fig.suptitle('The source posterior assigns poorer scores to these selected true models',x=.04,y=.96,ha='left',fontsize=18,fontweight='bold');fig.text(.04,.895,'PRIMARY SUPPORTED: posterior logloss exceeds uniform-candidate logloss by 0.81146 bits.',fontsize=12.5,fontweight='bold');fig.text(.04,.82,'All 1,067 examined feasible eight-positive checkpoints; 439 distinct remaining sets.\nThe 54 infeasible cases and 287 earlier successes remain separate context, outside the diagnostic mean.',fontsize=11.2,linespacing=1.5)
    u=float(s['primary']['mean_uniform_logloss_bits']);p=float(s['primary']['mean_posterior_logloss_bits']);loss.bar([0,1],[u,p],color=['#5c8d82','#b1745c'],width=.62);loss.set_xticks([0,1],['Uniform remaining\ncandidates','Source-posterior\nweights']);loss.set_ylabel('Mean true-model logloss, bits (lower is better)');loss.set_ylim(0,4);loss.set_title('Guarded primary threshold: ≥0.25 excess bits',fontsize=11.5,pad=14)
    for i,v in enumerate([u,p]):loss.text(i,v+.09,f'{v:.4f}',ha='center')
    nonempty=[r for r in bins if r['cases']];x=[number(r['mean_confidence']) for r in nonempty];y=[number(r['MAP_lex_accuracy']) for r in nonempty];reliability.plot([0,1],[0,1],ls='--',lw=1,color='#888');reliability.plot(x,y,lw=1,color='#6c8b9f',alpha=.5);reliability.scatter(x,y,s=[22+r['cases'] for r in nonempty],color='#527c99',alpha=.8,edgecolor='white')
    for a,b,r in zip(x,y,nonempty):reliability.text(a,b+.04,str(r['cases']),ha='center',fontsize=9.4)
    reliability.set_xlim(0,1);reliability.set_ylim(0,1);reliability.set_aspect('equal');reliability.set_xlabel('Mean top-1 posterior confidence within fixed bin');reliability.set_ylabel('Observed lex-MAP accuracy in selected cases');reliability.set_title('Descriptive reliability; labels are case counts',fontsize=11.5,pad=14)
    for ax in [loss,reliability]:ax.grid(alpha=.12);ax.set_axisbelow(True)
    confidence=100*number(s['descriptive_MAP_mean_confidence']);accuracy=100*number(s['descriptive_MAP_lex_accuracy']);gap=100*number(s['descriptive_MAP_overconfidence']);fig.text(.04,.235,f'Mean top-1 confidence {confidence:.2f}% versus observed accuracy {accuracy:.2f}%: gap {gap:.2f} percentage points.',fontsize=11.8,fontweight='bold');fig.text(.04,.165,'This is a finite diagnostic on selected, already examined truths. It does not estimate population calibration.\nAll ten fixed bins retained; [0, 0.1) is empty. MAP ties use the smallest candidate ID; rank intervals are preserved.',fontsize=11,linespacing=1.5);fig.text(.04,.09,'50-digit Decimal natural logs, independently checked through a score-based formula.\nSupport uses ≥0.25 + 10⁻¹² bits; within the numerical guard would be inconclusive. No diagnostic-selected exclusions.',fontsize=10.8,color='#444',linespacing=1.5);fig.text(.04,.03,f"BP AdamW: not measured. Zero new source draws or membership labels. {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s.",fontsize=10.6,color='#444')
    for ext in ['png','svg']:fig.savefig(PUBLIC/f'selected-posterior-diagnostic.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [path,bpath,Path(__file__)]},output_sha256={f'selected-posterior-diagnostic.{ext}':hashlib.sha256((PUBLIC/f'selected-posterior-diagnostic.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
