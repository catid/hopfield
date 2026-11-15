"""Plot only observed CPU gate expenditure; no unavailable speed bars."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3af'
def main():
    path=PUBLIC/'summary.json';s=json.loads(path.read_text());assert s['status']=='validated_failure_audit';a,b=s['family_results'];g=s['compilation_gate']
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3af-public'})
    fig,ax=plt.subplots(figsize=(12,6.2));fig.subplots_adjust(left=.15,right=.96,bottom=.37,top=.70)
    fig.suptitle('N15 exact compilation: the shared completeness gate timed out',x=.055,y=.95,ha='left',fontsize=17,fontweight='bold')
    fig.text(.055,.86,'Speed hypothesis: INCONCLUSIVE / NOT RUN · no external geometry gate, target cohorts or GPU jobs',fontsize=11.8,fontweight='bold')
    v=[a['complete_compile_cpu_seconds'],s['uncompleted_family_observed_cpu_remainder']];colors=['#287f77','#d7a044'];left=0
    for value,color,label in zip(v,colors,['Zebra: complete model-only cache','(2,5): incomplete + shared overhead']):
        ax.barh([0],[value],left=left,height=.48,color=color);ax.text(left+value/2,0,f'{value:.3f} CPU s',ha='center',va='center',fontweight='bold',color='white' if color==colors[0] else '#222');left+=value
    ax.axvline(120,color='#a43838',ls='--',lw=1.8);ax.text(120,.35,'120 CPU-s ceiling',ha='right',color='#a43838');ax.set_xlim(0,126);ax.set_ylim(-.6,.7);ax.set_yticks([0],['Two-family\nattempt']);ax.set_xlabel('Observed aggregate CPU seconds');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor=colors[0],label='Verified model-only: 74,436 states'),Patch(facecolor=colors[1],label='Provisional forward: 125,864 states')],loc='lower left',bbox_to_anchor=(.05,.25),ncol=2,frameon=False,fontsize=10.5)
    fig.text(.055,.195,f"Zebra wall-time breakdown: forward {a['primary_seconds']:.3f}s; independent reverse {a['independent_seconds']:.3f}s; full compilation {a['complete_compile_wall_seconds']:.3f}s.",fontsize=10.8)
    fig.text(.055,.13,f"Second-family forward: {b['primary_seconds']:.3f}s; reverse unfinished. Total gate wall time: {g['wall_seconds']:.3f}s.\nThe remaining CPU segment includes unfinished work and shared overhead, not a completed compilation.",fontsize=10.5,linespacing=1.5)
    fig.text(.055,.04,'BP AdamW source models unchanged. No partial-cache substitution or budget extension. Speed, accuracy and amortization are unmeasured.',fontsize=10.1,color='#444')
    for ext in ['png','svg']:fig.savefig(PUBLIC/f'compilation-gate.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [path,Path(__file__)]},output_sha256={f'compilation-gate.{ext}':hashlib.sha256((PUBLIC/f'compilation-gate.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])
if __name__=='__main__':main()
