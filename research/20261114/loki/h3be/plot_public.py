"""Show exact stratum denominators, integer thresholds and query-label costs."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3be'


def main():
    paths=[PUBLIC/'summary.json',PUBLIC/'runtime-parity.json'];s=json.loads(paths[0].read_text());rt=json.loads(paths[1].read_text());assert s['status']=='validated'
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3be-public'})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(14,8.8));fig.subplots_adjust(left=.08,right=.95,bottom=.30,top=.70,wspace=.30)
    fig.suptitle('MAP-guided identification passes full supplied-truth stress',x=.05,y=.95,ha='left',fontsize=18,fontweight='bold')
    fig.text(.05,.875,'SUPPORTED in every stratum: 1,489 / 1,511 exact models; 22 unresolved and 30 ineligible retained.',fontsize=12,fontweight='bold')
    fig.text(.05,.80,'One new sixteen-positive N12 bank per eligible truth; unchanged source prior and at most four binary N13 queries.\nKnown finite populations and candidate predictions; one bank does not establish per-model reliability.',fontsize=11.3,linespacing=1.5)
    for i,row in enumerate(s['strata']):
        fraction=row['identified']/row['eligible_truths'];required=row['required_exact']/row['eligible_truths'];ax.bar(i,fraction,width=.6,color='#26796b');ax.bar(i,1-fraction,bottom=fraction,width=.6,color='#d9b1a8');ax.plot([i-.35,i+.35],[required,required],color='#a14836',lw=2.4,label='Exact integer threshold' if i==0 else None);ax.text(i,1.045,f"{row['identified']:,}/{row['eligible_truths']:,}",ha='center',fontweight='bold');ax.text(i,.09,f"need {row['required_exact']:,}",ha='center',color='white',fontsize=10)
    ax.set_xticks([0,1,2],['Zero orbits\n1 truth','One orbit\n55 truths','Two orbits\n1,455 truths']);ax.set_ylim(0,1.16);ax.set_yticks([0,.25,.5,.75,1],['0%','25%','50%','75%','100%']);ax.set_ylabel('Exact unique identification');ax.set_title('Exact counts; red marks show required totals',fontweight='bold',fontsize=11.2,pad=20);ax.grid(axis='y',alpha=.14);ax.set_axisbelow(True)
    counts=[sum(r['eligible'] and r['membership_labels_used']==q for r in s['results']) for q in [1,2,3,4]];bx.bar([1,2,3,4],counts,color='#557d9a',width=.65)
    for q,count in zip([1,2,3,4],counts):bx.text(q,count+28,f'{count:,}',ha='center',fontsize=11)
    bx.set_xticks([1,2,3,4]);bx.set_ylim(0,1150);bx.set_xlabel('Binary membership labels used');bx.set_ylabel('Truth cases');bx.set_title('3,626 query labels in total',fontweight='bold',fontsize=12,pad=20);bx.grid(axis='y',alpha=.14);bx.set_axisbelow(True)
    fig.text(.05,.21,'Every one of the 22 unresolved models retains 2–9 candidates after four labels; no case is discarded.',fontsize=11.5,fontweight='bold')
    fig.text(.05,.14,'The k=0 stratum is a single case. All thirty M12<16 models remain ineligible, without draws or queries.\nNo fresh balanced/random control is claimed; BP AdamW is not measured in this exact query study.',fontsize=11,linespacing=1.5)
    fig.text(.05,.055,f"All 278 earlier selector decisions match exactly. First eight requests: {rt['first8_one_shot_wall_seconds']:.4f} s one-shot / {rt['first8_warm_persistent_wall_seconds']:.4f} s warm persistent.\nStudy: {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s. Full literal census inherited, fresh graph/filter/score parity.",fontsize=10.8,color='#444',linespacing=1.5)
    for ext in ['png','svg']:fig.savefig(PUBLIC/f'full-truth-class.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths+[Path(__file__)]},output_sha256={f'full-truth-class.{ext}':hashlib.sha256((PUBLIC/f'full-truth-class.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
