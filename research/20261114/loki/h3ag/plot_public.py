"""Tracked N15 quality and full-cost speed result, with failed threshold explicit."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3ag'
def main():
    path=PUBLIC/'summary.json';s=json.loads(path.read_text());assert s['status']=='validated';plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3ag-public'})
    fig,axes=plt.subplots(1,2,figsize=(14,8));fig.subplots_adjust(left=.07,right=.97,bottom=.36,top=.76,wspace=.28)
    fig.suptitle('Compiled verification completes N15; the full-cost 10× target fails',x=.07,y=.96,ha='left',fontsize=17,fontweight='bold')
    fig.text(.07,.895,'Two-family ≥10× hypothesis: FALSIFIED · correctness/completeness gates pass',fontsize=12.1,fontweight='bold',color='#973b36')
    fig.text(.07,.84,'Same frozen BP AdamW models and GPU kernels; fresh 4,096-input cohorts; three seeds, 20 outputs per input',fontsize=10.9)
    colors=['#c44e52','#19876b','#8064a2'];labels=['Zebra (2,3)','Leaper (2,5)']
    for i,f in enumerate(s['family_results']):
        for k,r in enumerate(f['rows']):
            x=i+(k-1)*.25;v=r['certified_success_pct'];axes[0].bar(x,v,width=.23,color=colors[k],alpha=.87);axes[0].annotate(f'{v:.2f}',(x,v),xytext=(0,5),textcoords='offset points',ha='center',fontsize=8.6)
        cpu=f['CPU_denominator_seconds'];gpu=f['GPU_numerator_seconds'];axes[1].bar(i-.15,gpu,width=.27,color=colors[1],alpha=.87);axes[1].bar(i+.15,cpu,width=.27,color=colors[2],alpha=.87);axes[1].plot([i-.35,i+.35],[gpu/10,gpu/10],color=colors[0],ls='--',lw=1.5)
        axes[1].annotate(f'{gpu:.3f}s',(i-.15,gpu),xytext=(0,6),textcoords='offset points',ha='center',fontsize=9);axes[1].annotate(f'{cpu:.3f}s\n{f["speed_ratio"]:.2f}× faster',(i+.15,cpu),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9,fontweight='bold')
    axes[0].set_ylim(0,113);axes[0].set_ylabel('Certified valid completion (%)');axes[0].set_title('A   Correctness and completeness gate',loc='left',fontweight='bold',pad=12);axes[1].set_ylim(0,97);axes[1].set_ylabel('Full seconds for three batches');axes[1].set_title('B   Compilation and prerequisites included',loc='left',fontweight='bold',pad=12)
    for ax in axes:ax.set_xlim(-.5,1.5);ax.set_xticks([0,1],labels);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor=colors[0],label='BP AdamW, arbitrary GPU'),Patch(facecolor=colors[1],label='BP AdamW, permutation GPU'),Patch(facecolor=colors[2],label='BP AdamW, exact CPU compilation')],loc='lower left',bbox_to_anchor=(.065,.275),ncol=3,frameon=False,fontsize=10)
    fig.text(.07,.225,'Full shared build/parity/monitor cost (6.243s) is charged to EACH family CPU denominator.',fontsize=11,fontweight='bold')
    fig.text(.07,.165,'Dashed marks: GPU time divided by 10. Exact CPU bars remain above those targets.\nCPU includes complete forward + compiled reverse enumeration, dense certificates, cache save/load and three query calls.',fontsize=10.3,linespacing=1.5)
    fig.text(.07,.09,f"All integer certificates pass; zero wrong stops. Complete caches: 74,436 / 125,864 states. GPU-worker cost: {s['total_GPU_worker_seconds']:.2f}s.\nProjected ≥10× reuse thresholds are 9 / 18 batches, but only THREE batches were measured: the extrapolation does not rescue this result.",fontsize=10.2,color='#444',linespacing=1.5)
    fig.text(.07,.03,'Fixed N15 models; inherited N13 caps are not newly time-matched. CPU/GPU hardware timing is not equal FLOPs or a single-query claim.',fontsize=10,color='#444')
    for ext in ['png','svg']:fig.savefig(PUBLIC/f'compiled-verifier.{ext}',dpi=180,facecolor='white',metadata={'Date':None}if ext=='svg'else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [path,Path(__file__)]},output_sha256={f'compiled-verifier.{ext}':hashlib.sha256((PUBLIC/f'compiled-verifier.{ext}').read_bytes()).hexdigest()for ext in ['png','svg']});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])
if __name__=='__main__':main()
