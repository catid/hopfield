"""Plot scoped certificate speedups from public paired-trial results."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3ah'
def main():
    path=PUBLIC/'summary.json';s=json.loads(path.read_text());assert s['status']=='validated';plt.rcParams.update({'font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3ah-public'})
    fig,axes=plt.subplots(1,2,figsize=(13.5,7.4));fig.subplots_adjust(left=.075,right=.97,bottom=.34,top=.75,wspace=.27)
    fig.suptitle('Exact occupied-pair certificates exceed the 20× speed target',x=.065,y=.95,ha='left',fontsize=17,fontweight='bold');fig.text(.065,.875,'All eight comparisons: SUPPORTED · every integer count agrees, including invalid states',fontsize=12,fontweight='bold');fig.text(.065,.82,'Frozen BP AdamW N15 models; three alternating paired trials; representation conversion charged',fontsize=10.9)
    colors=['#287f77','#8064a2']
    for ax,dataset,title in zip(axes,['complete_cache','arbitrary_cardinality15'],['A   Complete verified model caches','B   Arbitrary cardinality-15 states']):
        for bank in [0,1]:
            for k,input_path in enumerate(['available_columns_or_indices','full_board']):
                r=next(r for r in s['rows']if r['bank']==bank and r['dataset']==dataset and r['input_path']==input_path);x=bank+(k-.5)*.30;v=r['speed_ratio'];ax.bar(x,v,width=.28,color=colors[k],alpha=.9);ax.annotate(f'{v:.1f}×',(x,v),xytext=(0,5),textcoords='offset points',ha='center',fontsize=10,fontweight='bold')
        ax.axhline(20,color='#b34b43',ls='--',lw=1.5);ax.text(1.44,23,'20× target',ha='right',fontsize=9,color='#a33b35');ax.set_ylim(0,235);ax.set_xlim(-.5,1.5);ax.set_xticks([0,1],['Zebra (2,3)','Leaper (2,5)']);ax.set_ylabel('Dense median time / occupied-pair median time');ax.set_title(title,loc='left',fontweight='bold',pad=14);ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor=colors[0],label='Available row columns / occupied indices'),Patch(facecolor=colors[1],label='Full boards: extraction included')],loc='lower left',bbox_to_anchor=(.06,.25),ncol=2,frameon=False,fontsize=10.5)
    fig.text(.065,.20,'25,200 dense candidate pairs → 105 occupied pairs; both compute the same exact integer count.',fontsize=11,fontweight='bold');fig.text(.065,.14,'Complete caches: 74,436 / 125,864 states. Arbitrary grids: 8,195 states per model, all invalid, with conflicts up to 105.\n216,690 model–state pairs, 24 paired comparisons, 48 method calls; all inputs, counts and paired times retained.',fontsize=10.6,linespacing=1.5);fig.text(.065,.075,f"Whole experiment: {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s. No GPU work, new puzzle cohort or refit.",fontsize=10.5);fig.text(.065,.025,'CPU certificate microbenchmark only. This does not revise H3ag’s failed full-cost 10× compiler/GPU comparison.',fontsize=10.3,color='#444')
    for ext in ['png','svg']:fig.savefig(PUBLIC/f'occupied-pair-certificate.{ext}',dpi=180,facecolor='white',metadata={'Date':None}if ext=='svg'else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [path,Path(__file__)]},output_sha256={f'occupied-pair-certificate.{ext}':hashlib.sha256((PUBLIC/f'occupied-pair-certificate.{ext}').read_bytes()).hexdigest()for ext in ['png','svg']});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])
if __name__=='__main__':main()
