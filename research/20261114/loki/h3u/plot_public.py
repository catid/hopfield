"""Tracked H3u outcomes; deterministic publication figure."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[4];DEST=ROOT/'results/research/loki/h3u'
def main():
    path=DEST/'summary.json';s=json.loads(path.read_text());assert s['status']=='validated';p=s['primary']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3u-public'})
    fig,axes=plt.subplots(1,3,figsize=(15,7));fig.subplots_adjust(left=.055,right=.98,bottom=.39,top=.75,wspace=.32)
    fig.suptitle('Source-trained soft weights do not preserve certificate soundness',x=.055,y=.96,ha='left',fontsize=18,fontweight='bold')
    fig.text(.055,.89,f"Frozen matched BP AdamW hypothesis: {s['decision']}. Gain over filtered count: {p['accuracy_difference_pp']:+.2f} pp; required +5 pp.",fontsize=11.5,fontweight='bold')
    fig.text(.055,.83,'176 complete N12 source patterns · 4,096 fresh N13 inputs · 224 shared couplings + one unary in every arm',fontsize=11)
    arms=['bp-adamw-grouped','count-hybrid','exposure-filtered','supplied-rule-oracle'];labels=['BP AdamW\ngrouped','All source\nzeros','Exposure\nfiltered','Supplied\noracle'];colors=['#c44e52','#2878b5','#19876b','#8c5eb5']
    for k,arm in enumerate(arms):
        r=next(x for x in s['rows'] if x['arm']==arm)
        for offset,key,alpha,hatch in [(-.16,'original_accuracy_pct',.35,'//'),(.16,'certificate_first_accuracy_pct',.9,None)]:
            value=r[key];axes[0].bar(k+offset,value,width=.29,color=colors[k],alpha=alpha,hatch=hatch);axes[0].annotate(f'{value:.1f}',(k+offset,value),xytext=(0,5),textcoords='offset points',ha='center',fontsize=7)
        value=r['coverage_pct'];axes[1].bar(k,value,width=.62,color=colors[k],alpha=.87);axes[1].annotate(f'{value:.2f}',(k,value),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8.5)
        value=r['wrong_stops'];axes[2].bar(k,value,width=.62,color=colors[k],alpha=.87);axes[2].annotate(f'{value:,}',(k,value),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
    for ax,title in zip(axes,['A   Accuracy from 20 chains','B   Distinct-solution coverage','C   Wrong stopped chains']):
        ax.set_xticks(range(4),labels,fontsize=8);ax.set_xlim(-.55,3.55);ax.set_title(title,loc='left',fontweight='bold',pad=12);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    for ax in axes[:2]:ax.set_ylim(0,114);ax.set_ylabel('Percent')
    axes[2].set_ylim(0,max(r['wrong_stops'] for r in s['rows'])*1.22);axes[2].set_ylabel('Sum across three search seeds')
    fig.legend(handles=[Patch(facecolor='#777',alpha=.35,hatch='//',label='Original lowest-energy selection'),Patch(facecolor='#777',alpha=.9,label='Certificate-first selection')],loc='lower left',bbox_to_anchor=(.05,.29),ncol=2,frameon=False,fontsize=9)
    fit=s['fits']['bp-adamw-grouped'];fig.text(.055,.25,f"The source-fit gate passed: minimum source margin {fit['delta']:.6f}. Target wrong stops still violate the frozen requirement.",fontsize=10.8,fontweight='bold')
    fig.text(.055,.19,'Every source coefficient, unary and source-fitted stopping/temperature statistic transfers unchanged. The common q13 gate\nand source-inferred line extension remain supplied priors. No target calibration, optimizer tuning or post-failure repair.',fontsize=10,linespacing=1.5)
    fig.text(.055,.12,'All controls rerun under the same architecture and native source calibration. These settings differ from H3t’s q-scaled hard energies.\nThe oracle supplies target rules explicitly. Three seeds repeat search over one complete source universe; they are not fit replicas.',fontsize=9.5,color='#444',linespacing=1.5)
    num=all(d['numerical_gate_passed'] for d in s['output_diagnostics'])
    fig.text(.055,.055,f"{s['independent_saved_boards']:,} saved boards checked against exhaustive target solutions; unary-aware certificate capacities and float64 margins audited.\nFrozen numerical audit gate: {'passed' if num else 'failed'}. Exact-energy alternative selectors are diagnostics; reported selections use the frozen backend.",fontsize=9.3,color='#444',linespacing=1.5)
    for ext in ['png','svg']:fig.savefig(DEST/f'soft-transfer.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [path,Path(__file__)]},output_sha256={f'soft-transfer.{ext}':hashlib.sha256((DEST/f'soft-transfer.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']});(DEST/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt['output_sha256']))
if __name__=='__main__':main()
