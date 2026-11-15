"""Show the single eight-positive primary and its paired descriptive control."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3bm'


def main():
    path=PUBLIC/'summary.json';s=json.loads(path.read_text());rows=s['truth_results'];assert s['status']=='validated';plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3bm-public'})
    fig,ax=plt.subplots(figsize=(13,11));fig.subplots_adjust(left=.16,right=.92,bottom=.27,top=.73)
    fig.suptitle('Eight source positives fail the robustness target under the same query cap',x=.045,y=.96,ha='left',fontsize=17.8,fontweight='bold');fig.text(.045,.90,'PRIMARY FALSIFIED: 17/22 truths reach ≥61/64; five truths fall below the frozen requirement.',fontsize=12,fontweight='bold');fig.text(.045,.835,'Uniform eight-example subset of each sixteen-example bank; unchanged first-lex hybrid with at most eight target queries.\nEight positives: 1,354/1,408 exact. Paired sixteen-positive control: 1,408/1,408 exact.',fontsize=11.5,linespacing=1.5)
    for i,r in enumerate(rows):
        n=r['eight_positive_exact'];extra=r['sixteen_positive_exact']-n;ax.barh(i,n,color='#527f9e',height=.70,label='Eight positives: exact cases' if i==0 else None);ax.barh(i,extra,left=n,color='#76ad98',height=.70,label='Additional cases exact with sixteen' if i==0 else None);ax.text(n-.45,i,str(n),ha='right',va='center',color='white',fontsize=9.9,fontweight='bold' if n<61 else 'normal')
    ax.axvline(61,color='#a34f3e',ls='--',lw=1.2,zorder=0);ax.text(61,-1.05,'eight-positive primary: 61',ha='center',color='#a34f3e',fontsize=10);ax.set_yticks(range(len(rows)),[str(r['candidate']) for r in rows]);ax.invert_yaxis();ax.set_ylabel('True candidate ID');ax.set_xlim(0,68);ax.set_xticks([0,16,32,48,64]);ax.set_xlabel('Exact cases among the same 64 paired source banks');ax.grid(axis='x',alpha=.13);ax.set_axisbelow(True);ax.legend(loc='upper left',bbox_to_anchor=(0,1.135),frameon=False,ncol=2,fontsize=10.4)
    fig.text(.045,.205,'Eight positives require 8,302 membership calls; sixteen require 6,702. Less source evidence did not preserve accuracy.',fontsize=11.8,fontweight='bold');fig.text(.045,.14,'All 54 eight-positive failures retain >16 candidates after four MAP questions: four further binary questions cannot\nidentify every remaining model. This does not rule out a different first-stage policy under the same total cap.',fontsize=11.1,linespacing=1.5);fig.text(.045,.065,'22,528 positive-union appearances / 33,792 arm uses / 8,426 distinct source boards. 15,004 target calls,\nincluding 3,402 cross-arm reacquisitions. No exact bank duplicates; sixteen-positive results are descriptive control.',fontsize=10.7,color='#444',linespacing=1.5);fig.text(.045,.025,f"BP AdamW: not measured. Study {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; zero GPU work. Selected finite class.",fontsize=10.7,color='#444')
    for ext in ['png','svg']:fig.savefig(PUBLIC/f'eight-versus-sixteen.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [path,Path(__file__)]},output_sha256={f'eight-versus-sixteen.{ext}':hashlib.sha256((PUBLIC/f'eight-versus-sixteen.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
