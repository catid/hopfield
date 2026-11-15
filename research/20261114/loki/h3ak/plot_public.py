"""Fixed realized witness streams; primary64 endpoint distinguished from lower doses."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3ak'
def main():
    path=PUBLIC/'summary.json';s=json.loads(path.read_text());assert s['status']=='validated';plt.rcParams.update({'font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3ak-public'})
    fig,axes=plt.subplots(1,2,figsize=(14,7.5));fig.subplots_adjust(left=.075,right=.97,bottom=.36,top=.74,wspace=.28);fig.suptitle('64 positive witnesses per fit restore at least 95% exact N15 capacity',x=.065,y=.95,ha='left',fontsize=16.8,fontweight='bold');fig.text(.065,.875,'Four fixed acquired streams: SUPPORTED · no invalid zero-F states or missed true attack edges',fontsize=11.8,fontweight='bold');fig.text(.065,.815,'BP AdamW fits fixed; inherited finite exclusions deleted only when a selected valid N13 positive witnesses them',fontsize=10.7)
    colors=['#287f77','#7662a0','#bd7144','#497ba9'];doses=[0,1,8,32,64]
    for bank in range(4):
        rows=[r for r in s['rows']if r['bank']==bank];bydose={r['dose']:r for r in rows};ys=[rows[0]['baseline_H3aj_capacity_pct']]+[bydose[d]['capacity_pct']for d in doses[1:]];axes[0].plot(range(5),ys,'o-',color=colors[bank],lw=1.8,label=f'Bank {bank}');axes[0].scatter([4],[ys[-1]],s=65,color=colors[bank],edgecolor='white',zorder=5)
        r=bydose[64];axes[1].bar(bank,r['valid_state_retention_pct'],color=colors[bank],width=.65);axes[1].annotate(f"{r['valid_state_retention_pct']:.2f}%",(bank,r['valid_state_retention_pct']),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9.5)
    axes[0].axhline(95,color='#ae3f3a',ls='--',lw=1.5);axes[0].text(.05,95.3,'95% target',color='#ae3f3a',fontsize=9);axes[0].axvspan(3.75,4.25,color='#d7e6dc',alpha=.6,zorder=0);axes[0].set_xticks(range(5),['0\nPrior H3aj','1','8','32','64\nPRIMARY']);axes[0].set_xlim(-.2,4.3);axes[0].set_ylim(87.5,102);axes[0].set_ylabel('Exact capacity over 831,605 seen eligible keys (%)');axes[0].set_title('A   Lower doses descriptive; 64 is primary',loc='left',fontweight='bold',pad=14);axes[0].legend(frameon=False,loc='lower right',ncol=2,fontsize=9)
    axes[1].set_xticks(range(4),['Bank 0','Bank 1','Bank 2','Bank 3']);axes[1].set_ylim(0,112);axes[1].set_ylabel('True complete states retained after 64 positives (%)');axes[1].set_title('B   Capacity recovery is not always full graph recovery',loc='left',fontweight='bold',pad=14)
    for ax in axes:ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
    fig.text(.065,.255,'At 64 positives, capacities are 96.36%, 100%, 98.25%, 98.25%; false forbidden edges remain 182 / 0 / 91 / 91.',fontsize=11,fontweight='bold');fig.text(.065,.19,'Every deletion retains its exact source-positive witness. Repair cannot read the full source population or target truth/keys/scores.\nThe source-positive sampler is fixed before draws; all 4 banks and 16 bank-dose outcomes are retained.',fontsize=10.7,linespacing=1.5);fig.text(.065,.12,'Valid labels plus size-invariant pair rules imply deletion soundness. Recovery from a finite witness sample remains empirical.\nThe four realized streams do not establish a future-acquisition success probability.',fontsize=10.6,linespacing=1.5);fig.text(.065,.05,f"Whole stage: {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s. Same seen N15 universe: no untouched-test generalization claim.",fontsize=10.3,color='#444')
    for ext in ['png','svg']:fig.savefig(PUBLIC/f'positive-witness-repair.{ext}',dpi=180,facecolor='white',metadata={'Date':None}if ext=='svg'else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [path,Path(__file__)]},output_sha256={f'positive-witness-repair.{ext}':hashlib.sha256((PUBLIC/f'positive-witness-repair.{ext}').read_bytes()).hexdigest()for ext in ['png','svg']});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])
if __name__=='__main__':main()
