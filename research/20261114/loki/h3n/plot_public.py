"""Rebuild larger-source downward-transfer figure from tracked exports."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];DEST=ROOT/'results/research/loki/h3n'
def main():
    source=DEST/'summary.json';s=json.loads(source.read_text());assert s['status']=='validated'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3n-public'})
    arms=[None,'n12-full','n14-matched156','n14-full','true-graph-oracle'];colors=['#777777','#2878b5','#19876b','#ca8a04','#8c5eb5']
    labels=['BP AdamW\nnot run','All 156 N12\npatterns','156 sampled\nN14 patterns','All 5,180 N14\npatterns','Supplied\nN13 oracle']
    fig,axes=plt.subplots(1,2,figsize=(13,7));fig.subplots_adjust(left=.065,right=.98,bottom=.37,top=.75,wspace=.23)
    fig.suptitle('Larger source boards recover the smaller target graph',x=.065,y=.96,ha='left',fontsize=18,fontweight='bold')
    p=s['primary'];fig.text(.065,.89,f"Frozen full-N14 noninferiority hypothesis: {s['decision']}. N14-to-N13 downward transfer; margin −0.1 pp.",fontsize=11,fontweight='bold')
    fig.text(.065,.83,'4,096 fresh N13 superqueen inputs · 3 search seeds · all observed-zero offsets retained without pruning',fontsize=10.6)
    for k,arm in enumerate(arms):
        if arm is None:
            for ax in axes:ax.text(k,45,'Not measured\non superqueens',ha='center',va='center',color=colors[k],fontsize=8)
            continue
        row=next(r for r in s['rows'] if r['arm']==arm)
        assert row['original_accuracy_pct']==row['certificate_first_accuracy_pct']
        for ax,key in zip(axes,['certificate_first_accuracy_pct','coverage_pct']):
            value=row[key];ax.bar(k,value,width=.62,color=colors[k],alpha=.87,hatch='//' if arm in ['n14-full','true-graph-oracle'] else None)
            ax.annotate(f'{value:.2f}%',(k,value),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
    for ax,title in zip(axes,['A   Accuracy from 20 chains (N20)','B   Distinct-superqueen coverage']):
        ax.set_xticks(range(5),labels,fontsize=8.5);ax.set_xlim(-.55,4.55);ax.set_ylim(0,119)
        ax.set_ylabel('Percent');ax.set_title(title,loc='left',fontweight='bold',pad=12);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        ax.plot([2,4],[111,111],color='#333',lw=1);ax.text(3,112.5,'Exactly identical target energy',ha='center',fontsize=8)
    fig.text(.065,.285,'The same 156-label count gives 200 false exclusions from N12, but zero from this N14 subset.',fontsize=10.4,fontweight='bold')
    fig.text(.065,.225,'Both N14 sources supply witnesses against all 20 source-global N12 extra offsets. Their own remaining extra\nzero offsets lie outside the N13 displacement domain. None was removed using target labels.',fontsize=10,linespacing=1.5)
    fig.text(.065,.155,'Hatched bars are mapped from exact pre-inference parameter identity, not independent search replication.\nSix distinct model–seed searches ran; six duplicate jobs were avoided. Every stopped board is valid.',fontsize=9.5,linespacing=1.5)
    fig.text(.065,.085,'Original and certificate-first accuracies agree for every arm. Complete-source counts: N12 = 156, N14 = 5,180.\nOne fixed 156-pattern N14 subset: its success does not estimate reliability across independently sampled source banks.',fontsize=9,color='#444',linespacing=1.5)
    fig.text(.065,.025,'This is transfer to a smaller board with declared geometry, count and cardinality priors; it does not establish larger-target generalization.',fontsize=9,color='#444')
    for ext in ['png','svg']:fig.savefig(DEST/f'downward-transfer.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,Path(__file__)]},output_sha256={f'downward-transfer.{ext}':hashlib.sha256((DEST/f'downward-transfer.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']})
    (DEST/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt['output_sha256']))
if __name__=='__main__':main()
