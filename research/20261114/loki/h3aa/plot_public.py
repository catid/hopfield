"""Tracked initialization/proposal triangle with deterministic SVG metadata."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3aa'
def main():
    source=PUBLIC/'summary.json';s=json.loads(source.read_text());assert s['status']=='validated';p=s['primary'];cap=s['exact_zero_set_capacity']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3aa-public'})
    fig,axes=plt.subplots(1,3,figsize=(15,7.5));fig.subplots_adjust(left=.055,right=.98,bottom=.39,top=.75,wspace=.39)
    fig.suptitle('With shared starting states, proposal choice produces the large search gain',x=.055,y=.96,ha='left',fontsize=18,fontweight='bold')
    fig.text(.055,.89,f"Frozen hypothesis supported: {p['accuracy_difference_pp']:+.2f} pp certified gain; one-sided 95% endpoints [{p['accuracy_one_sided_95_lower_pp']:.2f}, {p['accuracy_one_sided_95_upper_pp']:.2f}].",fontsize=11.6,fontweight='bold')
    fig.text(.055,.83,'All arms use identical inherited BP AdamW weights and source temperature · primary arms share every initial-state byte',fontsize=11)
    labels=['Arbitrary init\nArbitrary swaps','Permutation init\nArbitrary swaps','Permutation init\nPermutation swaps'];colors=['#c78e93','#c44e52','#19876b']
    for k,row in enumerate(s['rows']):
        for shift,key,alpha,hatch in[(-.17,'certified_success_pct',.9,None),(.17,'certificate_first_accuracy_pct',.35,'//')]:
            v=row[key];axes[0].bar(k+shift,v,width=.3,color=colors[k],alpha=alpha,hatch=hatch);axes[0].annotate(f'{v:.2f}',(k+shift,v),xytext=(0,-14)if v>60 else(0,5),textcoords='offset points',ha='center',fontsize=8,bbox=dict(facecolor='white',edgecolor='none',pad=.1))
        axes[0].plot([k-.34,k+.34],[cap['certified_primary_ceiling_pct']]*2,ls=':',color='#333',lw=1.7)
        v=row['amortized_nanoseconds_per_proposal'];axes[2].bar(k,v,width=.6,color=colors[k]);axes[2].annotate(f'{v:.2f}',(k,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
    for ax in[axes[0],axes[2]]:ax.set_xticks(range(3),labels,fontsize=8.5)
    axes[0].set_ylim(0,113);axes[0].set_ylabel('Valid completion accuracy (%)');axes[0].set_title('A   Same model, three search recipes',loc='left',fontweight='bold',pad=12)
    for k,comp in enumerate(s['comparisons'][:2]):
        v=comp['accuracy_difference_pp'];lo=comp['accuracy_one_sided_95_lower_pp'];hi=comp['accuracy_one_sided_95_upper_pp'];color='#19876b'if k==0 else'#c44e52'
        axes[1].errorbar(v,k,xerr=[[v-lo],[hi-v]],fmt='o',color=color,capsize=4,ms=6);axes[1].annotate(f'{v:+.2f} pp',(v,k),xytext=(0,15),textcoords='offset points',ha='center',fontsize=9,fontweight='bold')
    axes[1].axvline(0,color='#aaa',lw=1);axes[1].axvline(50,color='#555',ls='--',lw=1);axes[1].text(50,.48,'+50 pp required\nfor primary',ha='center',fontsize=8.5,color='#555')
    axes[1].set_yticks([0,1],['Proposal effect\n(primary)','Initialization\n(descriptive)'],fontsize=9);axes[1].set_ylim(1.6,-.6);axes[1].set_xlim(-7,78);axes[1].set_xlabel('Paired certified-success difference (pp)');axes[1].set_title('B   Separate interventions',loc='left',fontweight='bold',pad=12)
    axes[2].set_ylim(0,55);axes[2].set_ylabel('Amortized search time / proposal (ns)');axes[2].set_title('C   Observed cost per proposal',loc='left',fontweight='bold',pad=12)
    for ax in axes:ax.grid(axis='x'if ax==axes[1]else'y',alpha=.18);ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor='#777',alpha=.9,label='Certified primary'),Patch(facecolor='#777',alpha=.35,hatch='//',label='Certificate-first with energy fallback')],loc='lower left',bbox_to_anchor=(.05,.28),ncol=2,frameon=False,fontsize=10)
    fig.text(.055,.23,'Primary: change proposals while holding permutation initialization fixed. Descriptive comparison: change initialization with arbitrary proposals fixed.',fontsize=10.6,fontweight='bold')
    fig.text(.055,.16,f"The shared support ceiling is {cap['certified_primary_ceiling_pct']:.2f}% (dotted); permutation proposals miss only 0 / 1 / 2 supported inputs across seeds.\nShared saved initial states and separate RNG streams prevent earlier search blocks from altering later initialization. All nine searches are fresh.",fontsize=10.1,linespacing=1.5)
    fig.text(.055,.085,'Permutation swaps change four cells and cost more per move. They use 100.36 s of measured search time versus 65.82 s for the shared-start\narbitrary kernel. Equal proposal caps do not imply equal FLOPs or wall time; this establishes a quality gain, not a speedup.',fontsize=9.8,color='#444',linespacing=1.5)
    fig.text(.055,.025,'737,280 states independently audited; zero wrong stops. Confidence bounds cluster three search seeds within each of 4,096 fresh inputs; one inherited fit.',fontsize=9.5,color='#444')
    for ext in['png','svg']:fig.savefig(PUBLIC/f'initialization-triangle.{ext}',dpi=180,facecolor='white',metadata={'Date':None}if ext=='svg'else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in[source,Path(__file__)]},output_sha256={f'initialization-triangle.{ext}':hashlib.sha256((PUBLIC/f'initialization-triangle.{ext}').read_bytes()).hexdigest()for ext in['png','svg']})
    (PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])
if __name__=='__main__':main()
