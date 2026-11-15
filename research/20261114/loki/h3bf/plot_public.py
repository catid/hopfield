"""Plot every selected truth against the frozen 48-of-64 endpoint."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3bf'


def main():
    path=PUBLIC/'summary.json';s=json.loads(path.read_text());assert s['status']=='validated';rows=s['truth_results'];plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3bf-public'})
    fig,ax=plt.subplots(figsize=(14,11));fig.subplots_adjust(left=.34,right=.94,bottom=.24,top=.76)
    fig.suptitle('Selected hard-tail failures persist across fresh source banks',x=.05,y=.96,ha='left',fontsize=18,fontweight='bold')
    fig.text(.05,.90,'FALSIFIED: only 6 of 22 selected truths meet the frozen requirement of 48 exact banks out of 64.',fontsize=12,fontweight='bold')
    fig.text(.05,.835,'All 22 truths are the previous full-class failures. This is a conditional follow-up, not a whole-class estimate.\nSame sixteen-positive sample size, source prior, MAP-guided policy, and four binary membership queries.',fontsize=11.2,linespacing=1.5)
    labels=[]
    for i,r in enumerate(rows):
        labels.append(f"{r['candidate']}: "+' + '.join(f'({a},{b})' for a,b in r['orbits']));color='#26796b' if r['identified']>=48 else '#b06450';ax.barh(i,r['identified'],height=.70,color=color);ax.text(r['identified']+.8,i,str(r['identified']),va='center',fontsize=10,color=color)
        if i%2==0:ax.axhspan(i-.48,i+.48,color='#777',alpha=.035)
    ax.axvline(48,color='#333',ls='--',lw=1.3);ax.text(48,-1.1,'required: 48',ha='center',fontsize=10);ax.set_yticks(range(len(rows)),labels);ax.invert_yaxis();ax.set_xlim(0,67);ax.set_xticks([0,16,32,48,64]);ax.set_xlabel('Exact unique identifications among 64 fresh source banks');ax.grid(axis='x',alpha=.14);ax.set_axisbelow(True)
    fig.text(.05,.17,'704/1,408 exact banks in the selected failure set; all 22 truths have 64 distinct unordered source banks.',fontsize=11.5,fontweight='bold')
    fig.text(.05,.11,'5,372 binary query labels; 704 cases remain ambiguous with 2–16 candidates. No extra questions or exclusions.\nFinite-bank threshold decisions do not prove a per-model population success probability above or below 75%.',fontsize=11,linespacing=1.5)
    fig.text(.05,.045,f"All 13,562 ledger records preserve hashes, order and timestamps; every query was fsynced before its answer.\nBP AdamW: not measured. Study {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s. Known census inherited with fresh parity.",fontsize=10.6,color='#444',linespacing=1.5)
    for ext in ['png','svg']:fig.savefig(PUBLIC/f'hard-tail-robustness.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [path,Path(__file__)]},output_sha256={f'hard-tail-robustness.{ext}':hashlib.sha256((PUBLIC/f'hard-tail-robustness.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
