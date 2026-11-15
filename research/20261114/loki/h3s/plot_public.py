"""Deterministic source-record replication diagnostic from tracked exports."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];DEST=ROOT/'results/research/loki/h3s'
def main():
    source=DEST/'summary.json';s=json.loads(source.read_text());assert s['status']=='validated'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3s-public'})
    fig,axes=plt.subplots(1,2,figsize=(12,6.5));fig.subplots_adjust(left=.075,right=.98,bottom=.35,top=.74,wspace=.27)
    fig.suptitle('Duplicating source records restores the false exclusions',x=.075,y=.95,ha='left',fontsize=18,fontweight='bold')
    fig.text(.075,.88,'Raw-record coefficient invariance: falsified. Explicit source deduplication: invariant at all tested copy counts.',fontsize=10.7,fontweight='bold')
    fig.text(.075,.81,'1×, 2× and 4× copies of the same 5,180 unique patterns · fixed exposure threshold · no new GPU search',fontsize=10.4)
    copies=[1,2,4]
    for variant,color,label,marker in [('raw-records','#ca8a04','Count raw records','o'),('source-deduplicated','#19876b','Deduplicate source patterns','s')]:
        vals=[next(r['expected_count'] for r in s['offset_scores'] if r['variant']==variant and r['copies']==c and r['dr']==12 and r['dc']==13) for c in copies]
        changed=[next(r['changed_unordered_pairs'] for r in s['rows'] if r['variant']==variant and r['copies']==c) for c in copies]
        for ax,v in zip(axes,[vals,changed]):ax.plot(copies,v,color=color,marker=marker,label=label,lw=2,markersize=7)
        for x,y in zip(copies,vals):axes[0].annotate(f'{y:.3f}',(x,y),xytext=(0,9 if variant=='raw-records' else -17),textcoords='offset points',ha='center',fontsize=9,color=color)
    threshold=s['rows'][0]['threshold'];axes[0].axhline(threshold,color='#b44',ls='--',label=f'Threshold {threshold:.3f}')
    axes[0].set_ylim(0,27);axes[0].set_ylabel('Exposure for each spurious offset');axes[0].set_title('A   Exposure changes without new information',loc='left',fontweight='bold',pad=12)
    axes[1].set_ylim(-2,29);axes[1].set_yticks([0,12,24]);axes[1].set_ylabel('Changed unordered target-pair coefficients');axes[1].set_title('B   Coefficient equality versus one copy',loc='left',fontweight='bold',pad=12)
    for ax in axes:
        ax.set_xscale('log',base=2);ax.set_xticks(copies,['1×','2×','4×']);ax.set_xlim(.9,4.45);ax.set_xlabel('Copies of each identical source pattern');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axes[0].legend(loc='upper left',fontsize=8,frameon=False);axes[1].legend(loc='upper left',fontsize=8,frameon=False)
    fig.text(.075,.265,'At 2× and 4×, raw-record fitting exactly recreates H3r’s unfiltered target model.',fontsize=11,fontweight='bold')
    fig.text(.075,.20,'Unique information and source marginals stay fixed, but E = N × sum(pᵢpⱼ) grows with the raw record count.\nDeduplication retains N = 5,180 and exactly preserves every target coefficient and solver parameter.',fontsize=10,linespacing=1.5)
    fig.text(.075,.13,'Mapped H3r accuracy is 99.943034% for the corrected model and 99.471029% for the restored unfiltered model.\nThose values are explanatory reuse under verified identity, not new experiments or independent replications. BP AdamW was not run.',fontsize=9.2,color='#444',linespacing=1.5)
    fig.text(.075,.055,'The vulnerability was analytically anticipated after H3r scores were known; copies 1/2/4 were frozen before this audit.\nDeduplication repairs this exact repeated-record effect, not arbitrary dependence, near-duplicates or general transfer failures.',fontsize=9.2,color='#444',linespacing=1.5)
    for ext in ['png','svg']:fig.savefig(DEST/f'record-replication.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,Path(__file__)]},output_sha256={f'record-replication.{ext}':hashlib.sha256((DEST/f'record-replication.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']})
    (DEST/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt['output_sha256']))
if __name__=='__main__':main()
