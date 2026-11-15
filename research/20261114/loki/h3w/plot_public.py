"""Tracked H3w comparison; deterministic publication figure."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[4];DEST=ROOT/'results/research/loki/h3w'
def main():
    source=DEST/'summary.json';s=json.loads(source.read_text());assert s['status']=='validated';p=s['primary'];cap=s['exact_zero_set_capacities']['bp-adamw-permutation']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3w-public'})
    fig,axes=plt.subplots(1,3,figsize=(15,7));fig.subplots_adjust(left=.055,right=.98,bottom=.39,top=.75,wspace=.32)
    fig.suptitle('Preserving rows and columns changes search with the same learned weights',x=.055,y=.96,ha='left',fontsize=17.3,fontweight='bold')
    fig.text(.055,.89,f"Frozen permutation-prior hypothesis: {s['decision']}. Certified gain {p['accuracy_difference_pp']:+.2f} pp; required +50 pp.",fontsize=11.5,fontweight='bold')
    fig.text(.055,.83,'Byte-identical inherited energies and source temperatures · fresh shared cohort · initialization and proposals change together',fontsize=10.6)
    labels=['BP AdamW\npermutation','BP AdamW\narbitrary-cell','Uniform F\npermutation','Oracle\npermutation'];colors=['#c44e52','#d99b9e','#19876b','#8c5eb5']
    for k,r in enumerate(s['rows']):
        for offset,key,alpha,hatch in [(-.16,'certified_success_pct',.9,None),(.16,'certificate_first_accuracy_pct',.35,'//')]:
            v=r[key];axes[0].bar(k+offset,v,width=.29,color=colors[k],alpha=alpha,hatch=hatch);axes[0].annotate(f'{v:.1f}',(k+offset,v),xytext=(0,-14)if v>70 else(0,5),textcoords='offset points',ha='center',fontsize=7.5,bbox=dict(facecolor='white',edgecolor='none',pad=.15),zorder=4)
        ceiling=s['exact_zero_set_capacities'][r['arm']]['certified_primary_ceiling_pct'];axes[0].plot([k-.32,k+.32],[ceiling,ceiling],color='#333',ls=':',lw=2)
        v=r['coverage_pct'];axes[1].bar(k,v,width=.62,color=colors[k],alpha=.87);axes[1].annotate(f'{v:.2f}',(k,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
        v=r['microseconds_per_proposal']*1000;axes[2].bar(k,v,width=.62,color=colors[k],alpha=.87);axes[2].annotate(f'{v:.1f}',(k,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
    for ax,title in zip(axes,['A   Certified and fallback accuracy','B   Distinct-solution coverage','C   Observed cost per proposal']):
        ax.set_xticks(range(4),labels,fontsize=8);ax.set_xlim(-.55,3.55);ax.set_title(title,loc='left',fontweight='bold',pad=12);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    for ax in axes[:2]:ax.set_ylim(0,114);ax.set_ylabel('Percent')
    axes[2].set_ylim(0,max(r['microseconds_per_proposal']*1000 for r in s['rows'])*1.25);axes[2].set_ylabel('Amortized GPU search time / proposal (ns)')
    fig.legend(handles=[Patch(facecolor='#777',alpha=.9,label='Certified primary: abstain if no zero-conflict solution'),Patch(facecolor='#777',alpha=.35,hatch='//',label='Certificate-first with best-energy fallback')],loc='lower left',bbox_to_anchor=(.05,.29),ncol=2,frameon=False,fontsize=9)
    fig.text(.055,.25,'The intervention adds uniform permutation initialization and symmetric column swaps between two nongiven rows.',fontsize=10.7,fontweight='bold')
    fig.text(.055,.19,f"All learned-support arms share the same {cap['certified_primary_ceiling_pct']:.2f}% certified-success ceiling (dotted). Strict integer certificates include given–given\nconflicts. Weights, source temperatures, support, 20 chains and 20,000-proposal caps remain frozen; no optimizer is refitted.",fontsize=10,linespacing=1.5)
    fig.text(.055,.12,'This measures a joint initialization/proposal/representation prior. It does not isolate a move-barrier cause. Permutation proposals\nchange four cells and are more expensive per proposal; equal proposal counts do not mean equal FLOPs or equal wall time.',fontsize=9.5,color='#444',linespacing=1.5)
    fig.text(.055,.055,f"{s['independent_saved_boards']:,} saved states audited against exhaustive target solutions; permutation states preserve every row, column and given.\nPrimary confidence bounds cluster all three search seeds within each of 4,096 fresh inputs, with one inherited fitted source model.",fontsize=9.3,color='#444',linespacing=1.5)
    for ext in ['png','svg']:fig.savefig(DEST/f'permutation-prior.{ext}',dpi=180,facecolor='white',metadata={'Date':None}if ext=='svg'else None)
    receipt=dict(input_sha256={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest()for x in [source,Path(__file__)]},output_sha256={f'permutation-prior.{ext}':hashlib.sha256((DEST/f'permutation-prior.{ext}').read_bytes()).hexdigest()for ext in ['png','svg']});(DEST/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt['output_sha256']))
if __name__=='__main__':main()
