"""Deterministic audit runtime figure: micro and full source job visibly distinct."""
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.hashsalt']='h2az-local-factor-audit'
ROOT=Path(__file__).resolve().parents[3];SRC=ROOT/'results/research/work/local-factor-audit';OUT=ROOT/'results/research/work/local-factor-audit';OUT.mkdir(exist_ok=True);micro=json.loads((SRC/'candidate-benchmark.json').read_text());full={m:json.loads((SRC/f'full-job-{m}.json').read_text()) for m in ['full_grid','local_factors']};cmp=json.loads((SRC/'full-job-comparison.json').read_text());fig,axes=plt.subplots(1,2,figsize=(10,5.2));labels=['Full-grid factors','Local affected factors'];colors=['#778899','#137c8b']
sets=[[statistics.median(r[m]['process_seconds'] for r in micro['rows']) for m in ['full_grid','local_factors']],[full[m]['wall_seconds'] for m in ['full_grid','local_factors']]]
for ax,values,title,ylabel in zip(axes,sets,['512 fixed candidate deletions','Complete fixed source audit job'],['Median process seconds (3 paired repeats)','Wall seconds (one matched pair)']):
 ax.bar(range(2),values,color=colors,width=.65);ax.set_xticks(range(2),labels);ax.set_ylim(0,max(values)*1.22);ax.set_ylabel(ylabel);ax.set_title(title,loc='left',fontsize=12);ax.spines[['top','right']].set_visible(False)
 for i,v in enumerate(values):ax.text(i,v+max(values)*.035,f'{v:.3f}s',ha='center',fontsize=11)
axes[0].text(.97,.94,f'{micro["median_paired_process_speedup"]:.2f}× paired process speedup',transform=axes[0].transAxes,ha='right',fontsize=10);axes[1].text(.97,.94,f'{cmp["wall_speedup"]:.2f}× complete-job speedup',transform=axes[1].transAxes,ha='right',fontsize=10);fig.suptitle('Independent local-factor audit: same objective checks, less work',x=.02,ha='left',fontsize=14);fig.text(.02,.025,'Work CPU 0; no inference rerun. Original H2ay 300-second full-audit timeout remains recorded. Complete-job audit fields agree.',fontsize=9);fig.tight_layout(rect=[0,.07,1,.94]);fig.savefig(OUT/'local-factor-audit.png',dpi=160,metadata={'Software':'Matplotlib'});fig.savefig(OUT/'local-factor-audit.svg',metadata={'Date':None});plt.close(fig)
