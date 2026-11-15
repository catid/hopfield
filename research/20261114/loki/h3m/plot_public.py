"""Rebuild the mixed-size superqueen figure from tracked exports only."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];DEST=ROOT/'results/research/loki/h3m'
def main():
    source=DEST/'summary.json';s=json.loads(source.read_text());assert s['status']=='validated'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3m-public'})
    arms=[None,'n12-only','multisize','n12-full','true-graph-oracle'];colors=['#777777','#2878b5','#19876b','#ca8a04','#8c5eb5']
    labels=['BP AdamW\nnot run','128 N12\npatterns','128 N12 +\nall 44 N11','All 156 N12\npatterns','Supplied\noracle']
    fig,axes=plt.subplots(1,2,figsize=(13,7));fig.subplots_adjust(left=.065,right=.98,bottom=.37,top=.75,wspace=.23)
    fig.suptitle('Adding a smaller source size leaves the failure unchanged',x=.065,y=.96,ha='left',fontsize=18,fontweight='bold')
    p=s['primary'];fig.text(.065,.89,f"Frozen ≥10 pp gain hypothesis: {s['decision']}. Mixed-minus-N12 gain = {p['accuracy_difference_pp']:.1f} pp (exact model identity).",fontsize=11,fontweight='bold')
    fig.text(.065,.83,'4,096 fresh N13 superqueen inputs · 3 search seeds · all observed-zero offsets retained without pruning',fontsize=10.6)
    for k,arm in enumerate(arms):
        if arm is None:
            for ax in axes:ax.text(k,45,'Not measured\non superqueens',ha='center',va='center',color=colors[k],fontsize=8)
            continue
        row=next(r for r in s['rows'] if r['arm']==arm)
        assert row['original_accuracy_pct']==row['certificate_first_accuracy_pct']
        for ax,key in zip(axes,['certificate_first_accuracy_pct','coverage_pct']):
            value=row[key];ax.bar(k,value,width=.62,color=colors[k],alpha=.87,hatch='//' if arm=='multisize' else None)
            ax.annotate(f'{value:.2f}%',(k,value),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
    for ax,title in zip(axes,['A   N20 superqueen accuracy','B   Distinct-superqueen coverage']):
        ax.set_xticks(range(5),labels,fontsize=8.5);ax.set_xlim(-.55,4.55);ax.set_ylim(0,113)
        ax.set_ylabel('Percent');ax.set_title(title,loc='left',fontweight='bold',pad=12);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        ax.plot([1,2],[106,106],color='#333',lw=1);ax.text(1.5,107.5,'Exactly identical energy',ha='center',fontsize=8)
    fig.text(.065,.285,'All 44 N11 solutions supply zero witnesses against the 69 zero-count offsets in the N12 bank.',fontsize=10.6,fontweight='bold')
    fig.text(.065,.225,'Mixed-size and base models each retain 212 false target pair exclusions. Full N12 removes one sample-only\nzero offset, but retains 200 false exclusions caused by absences shared by every N12 solution.',fontsize=10,linespacing=1.5)
    fig.text(.065,.155,'The hatched mixed-size bars are mapped from exact pre-inference identity; they are not independent runs.\nNine distinct model–seed searches ran; three duplicate jobs were avoided. Every stopped board is valid.',fontsize=9.5,linespacing=1.5)
    fig.text(.065,.085,'Original and certificate-first N20 scores agree for every arm. The oracle supplies target rules explicitly.\nMixed source adds 44 labels; full N12 adds 28, so the descriptive extra-label control is not perfectly count matched.',fontsize=9,color='#444',linespacing=1.5)
    fig.text(.065,.025,'One fixed N12 bank, complete N11/N12 universes, exhaustive N13 superqueen scoring; no broad source-size guarantee.',fontsize=9,color='#444')
    for ext in ['png','svg']:fig.savefig(DEST/f'multisize-source.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,Path(__file__)]},output_sha256={f'multisize-source.{ext}':hashlib.sha256((DEST/f'multisize-source.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']})
    (DEST/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt['output_sha256']))
if __name__=='__main__':main()
