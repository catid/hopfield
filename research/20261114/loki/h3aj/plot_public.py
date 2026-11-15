"""Exact state retention and input capacity, keeping all four fitted banks visible."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3aj'
def main():
    path=PUBLIC/'summary.json';s=json.loads(path.read_text());assert s['status']=='validated';plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3aj-public'})
    fig,axes=plt.subplots(1,3,figsize=(16,7.6));fig.subplots_adjust(left=.06,right=.97,bottom=.37,top=.74,wspace=.30);fig.suptitle('Incomplete source banks leave an exact N15 capacity ceiling below 95%',x=.055,y=.95,ha='left',fontsize=17,fontweight='bold');fig.text(.055,.875,'All four fixed BP AdamW fits: FALSIFIED · complete eligible universe, not sampled GPU accuracy',fontsize=12,fontweight='bold',color='#963b36');fig.text(.055,.815,'N12 source banks: 128 / 176 positives each. Four fitted models, two full supports; all banks retained.',fontsize=11)
    colors=['#287f77','#bc7045']
    for i,r in enumerate(s['rows']):
        color=colors[r['support_group']]
        for ax,key in zip(axes[:2],['valid_state_retention_pct','capacity_pct']):
            value=r[key];ax.bar(i,value,color=color,width=.65);ax.annotate(f'{value:.2f}',(i,value),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
    axes[0].set_ylim(0,110);axes[0].set_ylabel('True complete states retained (%)');axes[0].set_title('A   Complete solution retention',loc='left',fontweight='bold',pad=14);axes[1].set_ylim(0,110);axes[1].axhline(95,color='#ad3636',ls='--',lw=1.4);axes[1].text(3.5,99.5,'95% target',ha='right',fontsize=9,color='#ad3636');axes[1].set_ylabel('Eligible inputs with a zero-F completion (%)');axes[1].set_title('B   Exact input capacity',loc='left',fontweight='bold',pad=14)
    for ax in axes[:2]:ax.set_xticks(range(4),['Bank 0','Bank 1','Bank 2','Bank 3']);ax.set_xlim(-.6,3.6)
    for r in s['support_results']:
        ys=[r['per_clue'][str(k)]['capacity_pct']for k in [1,2,3]];axes[2].plot([1,2,3],ys,'o-',color=colors[r['group']],lw=2,label='Banks '+','.join(map(str,r['banks'])));axes[2].annotate(f'{ys[-1]:.2f}%',(3,ys[-1]),xytext=(-5,7 if r['group']==0 else -15),textcoords='offset points',ha='right',fontsize=9,fontweight='bold')
    axes[2].set_ylim(86,103);axes[2].set_xlim(.8,3.2);axes[2].set_xticks([1,2,3],['1 clue','2 clues','3 clues']);axes[2].set_ylabel('Exact input capacity (%)');axes[2].set_title('C   More clues expose exclusions',loc='left',fontweight='bold',pad=14);axes[2].legend(frameon=False,loc='lower left',fontsize=9)
    for ax in axes:ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor=colors[0],label='BP AdamW banks 0 / 1 / 3: shared support'),Patch(facecolor=colors[1],label='BP AdamW bank 2: distinct support')],loc='lower left',bbox_to_anchor=(.05,.27),ncol=2,frameon=False,fontsize=10.7)
    fig.text(.055,.215,'26,960 true N15 solutions; 831,605 eligible clue keys after 7,260 prior-key exclusions.',fontsize=11.4,fontweight='bold');fig.text(.055,.15,'All zero-F states are valid. The supports add 364 / 416 false forbidden edges and miss no true attack edge.\nInput capacity counts a clue key if ANY valid completion survives; it can stay high while many complete solutions are excluded.',fontsize=10.8,linespacing=1.5);fig.text(.055,.075,f"Full stage: {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s. All universes, integer certificates and key histograms audited.",fontsize=10.5);fig.text(.055,.025,'Exact finite-universe endpoint for fixed source fits and supplied size-invariant rules; no bootstrap, search-performance or general scaling claim.',fontsize=10.3,color='#444')
    for ext in ['png','svg']:fig.savefig(PUBLIC/f'incomplete-bank-capacity.{ext}',dpi=180,facecolor='white',metadata={'Date':None}if ext=='svg'else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [path,Path(__file__)]},output_sha256={f'incomplete-bank-capacity.{ext}':hashlib.sha256((PUBLIC/f'incomplete-bank-capacity.{ext}').read_bytes()).hexdigest()for ext in ['png','svg']});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])
if __name__=='__main__':main()
