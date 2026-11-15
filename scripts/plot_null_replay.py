"""Known-null retention versus extra-edge cost on the same acquired masks."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];source='results/research/ripper/null_replay.json';s=json.loads((ROOT/source).read_text());assert s['status']=='validated'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'text.color':'#17243a','axes.labelcolor':'#273449','axes.edgecolor':'#b7c1cf','svg.hashsalt':'null-replay-v1'})
fig,axes=plt.subplots(1,2,figsize=(14,7.5));colors=['#a44330','#347e72','#4976a8'];labels=['Calibrated null','Exact supplied law','Nominal iid at 5%']
for index,group in enumerate(s['grouped']):
    rows=[r for r in s['rows']if r['arm']==group['arm']]
    for ax,key in [(axes[0],'missing_true_edges'),(axes[1],'extra_edges')]:
        values=np.array([r[key]for r in rows]);ax.bar(index,values.mean(),width=.6,color=colors[index],alpha=.85,zorder=3)
        ax.scatter(index+np.linspace(-.13,.13,32),values,color='#17243a',s=14,alpha=.55,zorder=4)
        ax.text(index,values.max()+(.15 if key=='missing_true_edges'else 22),f'{group[key]:,} total',ha='center',weight='bold')
for ax in axes:ax.set_xticks(range(3),labels);ax.grid(axis='y',alpha=.15)
axes[0].set(ylim=(0,5),ylabel='Missing true edges per bank');axes[0].set_title('All true edges retained: 11 / 32 → 32 / 32',loc='left',weight='bold',pad=14)
axes[1].set(ylim=(0,1300),ylabel='Retained extra edges per bank');axes[1].set_title('Greater retention also preserves many extras',loc='left',weight='bold',pad=14)
fig.suptitle('Correcting the equality null restores retention on fixed acquired evidence',x=.065,ha='left',fontsize=18,weight='bold')
fig.text(.065,.035,
    'All 32 H1ap masks and labels unchanged; q1=1/9, first-crossing bound 648000. Pairs are never reinstated after crossing; unexposed pairs remain retained.\n'
    'Exact supplied permutation-law null 9/1600 and nominal iid null 311/25600 both retain all true edges. Neither recovers any exact graph on these fixed masks.\n'
    'For a queried true pair, the mask precedes the independent corruption draw: exact-law LR increments have conditional mean 1; nominal iid increments have mean < 1.\n'
    'Final-only thresholding is different: it reinstates 966 / 2,385 / 2,332 pair-bank entries. No acquisition policy was rerun using the new nulls.\n'
    f'Pre-run synthetic indexing failure was corrected before outcome access; originals and lineage preserved. {s["runtime"]["aggregate_cpu_s"]:.2f} CPU-s; {s["runtime"]["wall_s"]:.2f}s wall. No GPU. BP AdamW outside scope.',
    fontsize=9.15,color='#526075',linespacing=1.5)
fig.subplots_adjust(left=.065,right=.98,bottom=.30,top=.82,wspace=.25)
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/null-replay.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(fig)
(ROOT/'assets/figures/null-replay-provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs_sha256={source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},matplotlib=matplotlib.__version__,numpy=np.__version__,scope=s['scope']),indent=2)+'\n')
