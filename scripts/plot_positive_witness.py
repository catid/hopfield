"""Fixed-evidence comparison of conservative and positive-witness retention."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];source='results/research/ripper/positive_witness.json'
s=json.loads((ROOT/source).read_text());assert s['status']=='validated'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'text.color':'#17243a','axes.labelcolor':'#273449','axes.edgecolor':'#b7c1cf','svg.hashsalt':'positive-witness-v1'})
fig,axes=plt.subplots(1,2,figsize=(12.8,7.1));labels=['Archived likelihood ratio','Positive equality witness'];colors=['#a47d36','#347e72']
for index,(group,prefix)in enumerate(zip(s['grouped'],['baseline','witness'])):
    values=np.array([r[prefix+'_extras']for r in s['rows']]);axes[0].bar(index,values.mean(),width=.6,color=colors[index],alpha=.85,zorder=3)
    axes[0].scatter(index+np.linspace(-.13,.13,32),values,color='#17243a',s=17,alpha=.6,zorder=4)
    axes[0].text(index,values.max()+.3,f'{group["extra_edges"]} total',ha='center',weight='bold')
    axes[1].bar(index,group['exact_graph_banks'],width=.6,color=colors[index],alpha=.85,zorder=3)
    axes[1].text(index,group['exact_graph_banks']+.6,f'{group["exact_graph_banks"]}/32',ha='center',weight='bold')
for ax in axes:ax.set_xticks([0,1],labels);ax.grid(axis='y',alpha=.15)
axes[0].set(ylim=(0,6),ylabel='Retained extra edges per source bank');axes[0].set_title('All 54 extras already had an equality witness',loc='left',weight='bold',pad=14)
axes[1].set(ylim=(0,35),ylabel='Banks with exact graph');axes[1].axhline(29,color='#ba7e25',ls='--',lw=1.4)
axes[1].set_title('Frozen target: at least 29 exact graphs',loc='left',weight='bold',pad=14)
fig.suptitle('Changing retention alone recovers every graph on the same evidence',x=.07,ha='left',fontsize=17,weight='bold')
fig.text(.07,.04,
    'Post-hoc study on all 32 H1an filtered streams: unchanged masks, observed labels and accepted rows. No acquisition trajectory was rerun.\n'
    'Positive-witness rule: remove a pair after any accepted equal-digit coquery; retain unexposed pairs. No peer geometry enters the rule.\n'
    'Accepted labels are clean under the explicitly assumed collapse law and supplied digit-balance prior. Zero true edges are removed in either arm.\n'
    'Independent reconstruction: all 35,520 observed rows and 103,680 pair-bank counts checked. The rule is unsafe under undetected label errors.\n'
    f'No new queries, source rows, solver inputs or GPU work. {s["runtime"]["aggregate_cpu_s"]:.2f} CPU-s; {s["runtime"]["wall_s"]:.2f}s wall. BP AdamW outside scope; not run.',
    fontsize=9.2,color='#526075',linespacing=1.5)
fig.subplots_adjust(left=.07,right=.98,bottom=.30,top=.82,wspace=.28)
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/positive-witness.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(fig)
(ROOT/'assets/figures/positive-witness-provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs_sha256={source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},matplotlib=matplotlib.__version__,numpy=np.__version__,scope=s['scope']),indent=2)+'\n')
