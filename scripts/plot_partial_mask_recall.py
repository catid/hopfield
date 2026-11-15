"""Public-data-only rendering of the frozen H4au recall intervention."""
from pathlib import Path
import hashlib, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/research/foureyes/partial_mask_recall/summary.json'
PROTOCOL=ROOT/'research/20261114/foureyes/partial-mask-recall.json'
PARENT=ROOT/'results/research/foureyes/corruption_censor/summary.json'
OUT=ROOT/'assets/figures'


def main():
    s=json.loads(DATA.read_text());p=json.loads(PROTOCOL.read_text());parent=json.loads(PARENT.read_text())
    assert s['status']=='validated' and s['primary']['decision']=='supported'
    groups=s['grouped'];x=[g['recall_percent'] for g in groups];y=[g['success'] for g in groups]
    assert x==p['recall_percent'] and all(g['unresolved']==0 for g in groups)
    baseline=sum(r['baseline_exact'] for r in parent['rows']);full=sum(r['classification']=='success' for r in parent['rows'])
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'h4au-partial-mask-recall'})
    fig,ax=plt.subplots(figsize=(12,6.5));fig.subplots_adjust(left=.09,right=.95,top=.73,bottom=.30)
    fig.text(.04,.95,'Missing corruption labels: 256/256 graphs still recovered',fontsize=19,weight='bold')
    fig.text(.04,.875,'At 90% nominal recall and perfect precision, the frozen ≥245/256 target passes.',fontsize=13)
    ax.plot(x,y,'o-',color='#256d8d',lw=2,markersize=7,label='New partial-mask LP fits')
    ax.scatter([0,100],[baseline,full],marker='s',color='#777777',s=60,label='Inherited 0% / 100% controls',zorder=4)
    ax.scatter([90],[s['primary']['lower']],s=160,facecolors='none',edgecolors='#b85838',linewidth=2,zorder=5,label='Sole primary: 90% recall')
    ax.axhline(245,color='#b85838',ls='--',lw=1.2)
    ax.text(4,240,'90% arm target: ≥245',color='#9c452a',fontsize=10,va='top')
    ax.annotate('129 / 256',(0,baseline),xytext=(8,4),textcoords='offset points',weight='bold')
    ax.annotate('251 / 256',(50,251),xytext=(-42,-25),textcoords='offset points',weight='bold')
    ax.text(86,268,'256 / 256 at every 75–99% arm',ha='center',fontsize=11,weight='bold')
    ax.set_xlim(-3,103);ax.set_ylim(110,280);ax.set_xticks([0,25,50,75,90,100]);ax.set_yticks([128,160,192,224,256])
    ax.set_xlabel('Nominal recall of supplied corruption locations (%)');ax.set_ylabel('Exact graphs / 256 examined banks')
    ax.legend(loc='lower right',frameon=False,fontsize=10)
    at90=[r for r in s['rows'] if r['recall_percent']==90]
    contradictions=sum(r['true_edges_nonzero_cost'] for r in at90)
    fig.text(.04,.16,f'At 90%: {contradictions:,} true edges have nonzero counts; none of the never-equal supports is exact.\nThe degree-constrained LP recovers all graphs despite those residual collisions.',fontsize=11.3,linespacing=1.5)
    fig.text(.04,.055,'Privileged masks with no false positives; same 256 banks, 600 rows, fixed inferred degrees.\nOther recall arms are diagnostics. No detector trained. BP AdamW outside this source intervention.',fontsize=10.6,color='#444',linespacing=1.5)
    outputs=[]
    for ext in ['png','svg']:
        path=OUT/f'partial-mask-recall.{ext}';fig.savefig(path,dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None);outputs.append(path)
    sources=[DATA,PROTOCOL,PARENT,Path(__file__)]
    receipt=dict(input_sha256={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in sources},output_sha256={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in outputs})
    (OUT/'partial-mask-recall-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt['output_sha256']))


if __name__=='__main__':
    main()
