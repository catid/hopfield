"""Public full compiler costs with duplicated shared prerequisites explicit."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3ai'
def main():
    path=PUBLIC/'summary.json';s=json.loads(path.read_text());assert s['status']=='validated';plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3ai-public'})
    fig,ax=plt.subplots(figsize=(12.5,7.2));fig.subplots_adjust(left=.08,right=.96,bottom=.36,top=.73);fig.suptitle('Occupied-pair certificates improve full N15 compilation',x=.065,y=.95,ha='left',fontsize=17,fontweight='bold');fig.text(.065,.87,'Both ≥1.25× targets: SUPPORTED · complete models and caches remain byte-identical',fontsize=12,fontweight='bold');fig.text(.065,.81,'One crossed timing pair per fixed BP AdamW family; full shared prerequisites charged to every arm',fontsize=10.8)
    colors=['#c44e52','#287f77'];sharedcolor='#b6b8bc'
    for i,f in enumerate(s['families']):
        for k,r in enumerate(f['rows']):
            x=i+(k-.5)*.3;ax.bar(x,r['compile_seconds'],width=.28,color=colors[k],alpha=.9);ax.bar(x,r['shared_cost_charged_seconds'],bottom=r['compile_seconds'],width=.28,color=sharedcolor);ax.annotate(f"{r['full_charged_seconds']:.3f}s",(x,r['full_charged_seconds']),xytext=(0,5),textcoords='offset points',ha='center',fontsize=10)
        ax.text(i,32,f"{f['speed_ratio']:.3f}× faster",ha='center',fontweight='bold',fontsize=12)
    ax.set_ylim(0,35);ax.set_xlim(-.55,1.55);ax.set_xticks([0,1],['Zebra: dense → occupied','Leaper (2,5): occupied → dense']);ax.set_ylabel('Complete charged compilation wall seconds');ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor=colors[0],label='BP AdamW, dense certificate'),Patch(facecolor=colors[1],label='BP AdamW, occupied-pair certificate'),Patch(facecolor=sharedcolor,label='Full shared build/parity/monitor')],loc='lower left',bbox_to_anchor=(.06,.265),ncol=3,frameon=False,fontsize=9.8)
    fig.text(.065,.215,f"Shared overhead: {s['shared_prerequisite_wall_seconds']:.3f}s added in full to all four arms; actual aggregate cost is counted once.",fontsize=10.8,fontweight='bold');fig.text(.065,.15,'Only certificate summation changes. Both arms retransfer, fully enumerate with both algorithms, verify, build the index and save.\nComplete cache sizes: 74,436 / 125,864 states. All integer certificates independently recomputed after the run.',fontsize=10.5,linespacing=1.5);fig.text(.065,.08,f"Whole experiment: {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s. One pair per family is a point estimate, not a timing confidence interval.",fontsize=10.2);fig.text(.065,.025,'Full compiler timing only: no new CPU/GPU comparison or puzzle cohort. H3ag’s failed three-batch 10× hypothesis is unchanged.',fontsize=10.1,color='#444')
    for ext in ['png','svg']:fig.savefig(PUBLIC/f'full-compiler-integration.{ext}',dpi=180,facecolor='white',metadata={'Date':None}if ext=='svg'else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [path,Path(__file__)]},output_sha256={f'full-compiler-integration.{ext}':hashlib.sha256((PUBLIC/f'full-compiler-integration.{ext}').read_bytes()).hexdigest()for ext in ['png','svg']});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])
if __name__=='__main__':main()
