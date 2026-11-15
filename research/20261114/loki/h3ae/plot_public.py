"""Validated model-only compilation quality and full scoped timing."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3ae'
def main():
    path=PUBLIC/'summary.json';s=json.loads(path.read_text());assert s['status']=='validated';families=s['family_results'];plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3ae-public'})
    fig,axes=plt.subplots(1,2,figsize=(14,8));fig.subplots_adjust(left=.07,right=.97,bottom=.36,top=.76,wspace=.28)
    fig.suptitle('Reusable fixed-N13 models: exact CPU compilation beats frozen GPU sampling',x=.07,y=.96,ha='left',fontsize=16.5,fontweight='bold')
    fig.text(.07,.895,f"Two-family 10× speed claim: {s['decision'].upper()} · compilation and independent completeness verification included",fontsize=11.9,fontweight='bold')
    fig.text(.07,.84,'Same frozen BP AdamW models; two fresh 4,096-input cohorts; three query/search seeds and 20 outputs per input',fontsize=10.9)
    colors=['#c44e52','#19876b','#8064a2'];labels=['Zebra (2,3)','Leaper (2,5)']
    for i,f in enumerate(families):
        for k,r in enumerate(f['rows']):
            x=i+(k-1)*.25;v=r['certified_success_pct'];axes[0].bar(x,v,width=.23,color=colors[k],alpha=.87);axes[0].annotate(f'{v:.2f}',(x,v),xytext=(0,5),textcoords='offset points',ha='center',fontsize=8.6)
        cpu=f['CPU_denominator_seconds'];gpu=f['GPU_numerator_seconds'];axes[1].bar(i-.15,gpu,width=.27,color=colors[1],alpha=.87);axes[1].bar(i+.15,cpu,width=.27,color=colors[2],alpha=.87);axes[1].plot([i-.35,i+.35],[gpu/10,gpu/10],color='#c44e52',ls='--',lw=1.5)
        axes[1].annotate(f'{gpu:.3f}s',(i-.15,gpu),xytext=(0,6),textcoords='offset points',ha='center',fontsize=9);axes[1].annotate(f'{cpu:.3f}s\n{f["speed_ratio"]:.2f}× faster',(i+.15,cpu),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9,fontweight='bold',bbox=dict(facecolor='white',edgecolor='none',pad=1.5))
    axes[0].set_ylim(0,113);axes[0].set_ylabel('Certified valid completion (%)');axes[0].set_title('A   Correctness and completeness gate',loc='left',fontweight='bold',pad=12);axes[1].set_yscale('log');axes[1].set_ylim(.5,130);axes[1].set_ylabel('Total seconds for three batches (log scale)');axes[1].set_title('B   Includes one full CPU compilation',loc='left',fontweight='bold',pad=12)
    for ax in axes:ax.set_xlim(-.5,1.5);ax.set_xticks([0,1],labels);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor=colors[0],label='BP AdamW, arbitrary GPU'),Patch(facecolor=colors[1],label='BP AdamW, permutation GPU'),Patch(facecolor=colors[2],label='BP AdamW, exact CPU compilation')],loc='lower left',bbox_to_anchor=(.065,.275),ncol=3,frameon=False,fontsize=10)
    fig.text(.07,.225,'CPU time = model checks + enumeration + independent proof + cache serialization/load + three query calls.',fontsize=10.8,fontweight='bold')
    fig.text(.07,.16,'The dashed timing marks are GPU time divided by 10. GPU numerator includes three complete synchronized solve calls.\nWarmups, scoring and record serialization are outside the compared solve calls; all GPU worker costs remain charged to the study.',fontsize=10.2,linespacing=1.5)
    fig.text(.07,.085,f"Full GPU-worker cost: {s['total_gpu_worker_seconds']:.2f}s. Cache sizes: 2,696 and 6,368 solutions. Query workers read clue-only files.\nThe exact quality result follows complete zero-F enumeration and validity; the measured speed threshold is the falsifiable endpoint.",fontsize=10.1,color='#444',linespacing=1.5)
    fig.text(.07,.03,'Fixed N13 models and three batches amortize one cache. This is hardware-specific timing, not equal FLOPs, single-query speed or a scaling guarantee.',fontsize=9.9,color='#444')
    for ext in['png','svg']:fig.savefig(PUBLIC/f'exact-compilation.{ext}',dpi=180,facecolor='white',metadata={'Date':None}if ext=='svg'else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in[path,Path(__file__)]},output_sha256={f'exact-compilation.{ext}':hashlib.sha256((PUBLIC/f'exact-compilation.{ext}').read_bytes()).hexdigest()for ext in['png','svg']});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])
if __name__=='__main__':main()
