"""Deterministic visualization of the retained ceiling falsification."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3an'
def main():
    path=PUBLIC/'summary.json';s=json.loads(path.read_text());assert s['status']=='validated';plt.rcParams.update({'font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3an-public'});fig,axes=plt.subplots(1,2,figsize=(14,7.5));fig.subplots_adjust(left=.075,right=.97,bottom=.35,top=.75,wspace=.31);fig.suptitle('New-family D4 benefit fails: every 128-positive fit is already exact',x=.065,y=.95,ha='left',fontsize=16.4,fontweight='bold');fig.text(.065,.879,'FALSIFIED: ≥10 pp gain fails in all 8 fits · reliability and zero-invalid-state criteria pass',fontsize=11.8,fontweight='bold');fig.text(.065,.819,'BP AdamW: four independent incomplete N12 banks per family; unchanged 4,000-update recipe',fontsize=11)
    for i in range(2):
        for shift,label,color in[(-.23,'Before acquisition','#7f8c8d'),(0,'Raw repair','#497ba9'),(.23,'D4 repair','#287f77')]:axes[0].bar(i+shift,100,width=.21,color=color,label=label if i==0 else None)
        axes[0].annotate('All 4 fits: 100%',(i,100),xytext=(0,6),textcoords='offset points',ha='center',fontsize=10,fontweight='bold')
    axes[0].set_xticks([0,1],['Zebra','(2,5) leaper']);axes[0].set_ylim(0,114);axes[0].set_ylabel('Exact capacity / acquisition success probability (%)');axes[0].set_title('A   Capacity and reliability already at their ceiling',loc='left',fontweight='bold',pad=14);axes[0].legend(frameon=False,loc='lower left',fontsize=9)
    for i in range(8):axes[1].scatter(i,0,s=55,color='#287f77');axes[1].annotate('0',(i,0),xytext=(0,6),textcoords='offset points',ha='center',fontsize=9)
    axes[1].axhline(10,color='#ae3f3a',ls='--',lw=1.4);axes[1].text(.05,10.35,'Frozen ≥10 pp target',color='#ae3f3a',fontsize=9);axes[1].set_ylim(-1.8,13);axes[1].set_xticks(range(8),['Z0','Z1','Z2','Z3','L0','L1','L2','L3']);axes[1].set_ylabel('Same-draw D4 reliability gain (percentage points)');axes[1].set_title('B   Every new fit retains its zero-gain outcome',loc='left',fontweight='bold',pad=14)
    for ax in axes:ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
    fig.text(.065,.255,'All 8 fits: exact target graph, zero false/missed edges, no observable repair classes, zero invalid states.',fontsize=11,fontweight='bold');fig.text(.065,.19,'Training: 128 of 472 / 1,304 N12 positives. Acquisition: 32 from each complement (344 / 1,176 boards).\nD4 adds an explicit invariance prior and 256 attempted correlated views. No target positives or target-driven selection.',fontsize=10.7,linespacing=1.5);fig.text(.065,.12,'Full seen N13 capacity, two complete supports; all 8 distinct fit records retained. No new GPU search.\nThe failed benefit endpoint is preserved; passing safety and reliability does not rescue the conjunctive hypothesis.',fontsize=10.7,linespacing=1.5);fig.text(.065,.05,f"Whole stage: {s['aggregate_CPU_seconds']:.3f} CPU s / {s['aggregate_GPU_worker_seconds']:.3f} GPU-worker s / {s['wall_seconds']:.3f} wall s. Both GPUs ran advancing fit queues.",fontsize=10.3,color='#444')
    for ext in('png','svg'):fig.savefig(PUBLIC/f'new-family-ceiling.{ext}',dpi=180,facecolor='white',metadata={'Date':None}if ext=='svg'else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in(path,Path(__file__))},output_sha256={f'new-family-ceiling.{ext}':hashlib.sha256((PUBLIC/f'new-family-ceiling.{ext}').read_bytes()).hexdigest()for ext in('png','svg')});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])
if __name__=='__main__':main()
