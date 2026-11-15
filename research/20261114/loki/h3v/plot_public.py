"""H3v publication figure reads tracked outcome/weight exports only."""
import csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[4];DEST=ROOT/'results/research/loki/h3v'
def main():
    source=DEST/'summary.json';weight_path=ROOT/'results/research/loki/h3u/source_offset_weights.csv';s=json.loads(source.read_text());weights=list(csv.DictReader(weight_path.open()));assert s['status']=='validated'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3v-public'})
    fig,axes=plt.subplots(1,3,figsize=(15,7));fig.subplots_adjust(left=.055,right=.98,bottom=.39,top=.75,wspace=.32)
    fig.suptitle('Uniform weights find more solutions on the same certified zero set',x=.055,y=.96,ha='left',fontsize=17.3,fontweight='bold')
    p=s['primary'];fig.text(.055,.89,f"Frozen learned-weight gain hypothesis: {s['decision']}. Certified success difference {p['accuracy_difference_pp']:+.2f} pp; required +5 pp.",fontsize=11.5,fontweight='bold')
    fig.text(.055,.83,'Same filtered support · positive weights only · fixed-q Metropolis swaps · 4,096 fresh N13 inputs × three search seeds',fontsize=10.8)
    colors=['#c44e52','#19876b','#8c5eb5'];labels=['BP AdamW weights\nprojected to support','Uniform weights\nsame support','Supplied\noracle']
    for k,r in enumerate(s['rows']):
        for offset,key,alpha,hatch in [(-.16,'certified_success_pct',.9,None),(.16,'certificate_first_accuracy_pct',.35,'//')]:
            value=r[key];axes[0].bar(k+offset,value,width=.29,color=colors[k],alpha=alpha,hatch=hatch);axes[0].annotate(f'{value:.1f}',(k+offset,value),xytext=(0,-14) if value>70 else (0,5),textcoords='offset points',ha='center',fontsize=8,bbox=dict(facecolor='white',edgecolor='none',pad=.15),zorder=4)
        cap=s['exact_zero_set_capacities'][r['arm']]['certified_primary_ceiling_pct'];axes[0].plot([k-.32,k+.32],[cap,cap],color='#333',ls=':',lw=2)
        value=r['coverage_pct'];axes[1].bar(k,value,width=.62,color=colors[k],alpha=.87);axes[1].annotate(f'{value:.2f}',(k,value),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
    for ax,title in zip(axes[:2],['A   Certified and fallback accuracy','B   Distinct-solution coverage']):
        ax.set_xticks(range(3),labels,fontsize=8);ax.set_xlim(-.55,2.55);ax.set_ylim(0,114);ax.set_ylabel('Percent');ax.set_title(title,loc='left',fontweight='bold',pad=12);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    unit=next(m['unit']for m in s['independent_preflight']['models']if m['arm']=='positive-bp-adamw-weights')
    offsets=[(0,1),(1,1),(1,3),(2,8),(2,11)];values=[float(next(r['bp_adamw_weight']for r in weights if(int(r['dr']),int(r['dc']))==o))/unit for o in offsets]
    axes[2].bar(range(5),values,color=['#c44e52']*3+['#ca8a04']*2,width=.65)
    for k,v in enumerate(values):axes[2].annotate(f'{v:.3f}',(k,v),xytext=(0,6),textcoords='offset points',ha='center',fontsize=8)
    axes[2].axhline(1,color='#19876b',ls='--',label='Uniform coefficients / unit');axes[2].set_xticks(range(5),['Row /\ncolumn','Diagonal','Camel','Extra\n(2,8)','Extra\n(2,11)'],fontsize=8);axes[2].set_ylim(0,2.1);axes[2].set_ylabel('Representative source coupling / source unit');axes[2].set_title('C   Unequal retained penalties',loc='left',fontweight='bold',pad=12);axes[2].legend(loc='upper right',fontsize=8,frameon=False);axes[2].grid(axis='y',alpha=.18);axes[2].set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor='#777',alpha=.9,label='Certified primary: abstain if no zero-conflict solution'),Patch(facecolor='#777',alpha=.35,hatch='//',label='Certificate-first with best-energy fallback')],loc='lower left',bbox_to_anchor=(.05,.29),ncol=2,frameon=False,fontsize=9)
    cap=s['exact_zero_set_capacities']['uniform-filtered'];fig.text(.055,.25,f"Both learned-support arms have zero wrong stops and the same {cap['certified_primary_ceiling_pct']:.2f}% certified-success ceiling (dotted).",fontsize=10.8,fontweight='bold')
    fig.text(.055,.19,'Positive coefficients change search, not the zero set. The unchanged support retains 308 false exclusions; 482 inputs have no\nvalid zero-conflict completion. All given–given conflicts count in this stricter certificate. The oracle supplies the true attack graph.',fontsize=10,linespacing=1.5)
    fig.text(.055,.12,'Temperature follows the frozen source-only geometric schedule from mean retained coefficient to 1% of that value.\nThe unequal coefficients are descriptive mechanism evidence; no causal barrier diagnosis or target-based tuning is claimed.',fontsize=9.5,color='#444',linespacing=1.5)
    fig.text(.055,.055,f"{s['independent_saved_boards']:,} saved boards audited; exact integer conflict certificates are independent of float energy tolerance.\nThis is a new fixed-cardinality search derivative with inherited weights, zero optimizer refits and one complete source universe.",fontsize=9.3,color='#444',linespacing=1.5)
    for ext in ['png','svg']:fig.savefig(DEST/f'fixed-support-search.{ext}',dpi=180,facecolor='white',metadata={'Date':None}if ext=='svg'else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [source,weight_path,Path(__file__)]},output_sha256={f'fixed-support-search.{ext}':hashlib.sha256((DEST/f'fixed-support-search.{ext}').read_bytes()).hexdigest()for ext in ['png','svg']});(DEST/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt['output_sha256']))
if __name__=='__main__':main()
