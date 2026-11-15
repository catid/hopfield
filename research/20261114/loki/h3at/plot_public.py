"""Show capacity loss after expanding the rule vocabulary, including all banks."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3at'


def main():
    paths=[PUBLIC/'summary.json',PUBLIC/'source-selection.json',ROOT/'results/research/loki/h3ar/summary.json'];s=json.loads(paths[0].read_text());source=json.loads(paths[1].read_text());narrow=json.loads(paths[2].read_text());assert s['status']==narrow['status']=='validated'
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3at-public'})
    fig,axes=plt.subplots(1,2,figsize=(14,7.8),gridspec_kw={'width_ratios':[1.6,1]});fig.subplots_adjust(left=.075,right=.96,bottom=.31,top=.75,wspace=.28)
    fig.suptitle('Expanding the supplied vocabulary causes finite-sample overfitting',x=.065,y=.95,ha='left',fontsize=17,fontweight='bold')
    fig.text(.065,.88,'FALSIFIED: only 2 / 8 ML models retain ≥95% capacity; all eight unique selections add a false orbit',fontsize=11.8,fontweight='bold')
    fig.text(.065,.818,'Same 16 raw positives per bank; 55 orbits and 1,541 models replace 10 orbits and 56 models; no BP fits',fontsize=10.9)
    details=[]
    for b in source['banks']:
        index=b['index'];ml=next(r for r in s['rows'] if r['index']==index and r['primary']);control=next(r for r in s['rows'] if r['index']==index and not r['primary']);old=next(r for r in narrow['rows'] if r['index']==index and r['primary']);assert old['capacity_pct']==100
        axes[0].bar(index-.17,ml['capacity_pct'],.32,color='#29677c',label='Expanded-vocabulary ML' if index==0 else None);axes[0].bar(index+.17,control['capacity_pct'],.32,color='#99a8ac',label='Consistency envelope' if index==0 else None);axes[0].text(index-.17,ml['capacity_pct']-5.5,f"{ml['capacity_pct']:.1f}",ha='center',fontsize=8.8,color='white')
        true=(2,3) if index<4 else (2,5);orbits=source['candidates'][ml['candidate']]['orbits'];extra=[tuple(o) for o in orbits if tuple(o)!=true];details.append([('Z' if index<4 else 'L')+str(index%4),str(extra[0]),str(ml['false_forbidden_edges'])])
    axes[0].axhline(100,color='#44816c',ls=':',lw=1.3,label='Prior narrow-vocabulary ML: 100%');axes[0].axhline(95,color='#ad4841',ls='--',lw=1.2);axes[0].set_ylim(0,113);axes[0].set_xticks(range(8),['Z0','Z1','Z2','Z3','L0','L1','L2','L3']);axes[0].set_ylabel('Exact valid-input capacity (%)');axes[0].set_title('A   Six banks fall below the frozen 95% threshold',loc='left',fontsize=11.4,fontweight='bold',pad=14);axes[0].grid(axis='y',alpha=.15);axes[0].set_axisbelow(True);fig.legend(*axes[0].get_legend_handles_labels(),frameon=False,loc='lower left',bbox_to_anchor=(.075,.215),fontsize=9.3,ncol=3)
    axes[1].axis('off');axes[1].set_title('B   Every ML selection retains a spurious rule',loc='left',fontsize=11.4,fontweight='bold',pad=14)
    table=axes[1].table(cellText=details,colLabels=['Bank','False orbit','False edges'],cellLoc='center',colLoc='center',bbox=[0,.03,1,.95]);table.auto_set_font_size(False);table.set_fontsize(10.9)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('#ddd');cell.set_linewidth(.5)
        if row==0:cell.set_facecolor('#e8eff0');cell.set_text_props(weight='bold')
    fig.text(.065,.17,'All selected models have zero invalid states and zero missed true edges, but exclude valid completions.',fontsize=11,fontweight='bold')
    fig.text(.065,.11,'Selected 0/1/2-orbit models improve capacity over the broad envelope in six banks; both coincide in L0 and L3.\nThe broad envelope equals the prior H3aq D4-unfiltered graph in every bank (post-hoc full-array identity check).',fontsize=10.7,linespacing=1.5)
    fig.text(.065,.045,f"All 1,541 source normalizers, 12,328 likelihoods and 12 complete N13 supports checked; no ties or invalid states discarded.\nStudy: {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s. Same seen banks and targets; adaptive prior-sensitivity follow-up.",fontsize=10.3,color='#444',linespacing=1.4)
    for ext in ('png','svg'):fig.savefig(PUBLIC/f'expanded-vocabulary.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[Path(__file__)]},output_sha256={f'expanded-vocabulary.{ext}':hashlib.sha256((PUBLIC/f'expanded-vocabulary.{ext}').read_bytes()).hexdigest() for ext in ('png','svg')});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
