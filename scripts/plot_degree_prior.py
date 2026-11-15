"""Explicit structural information, finite-bank recovery, and exact certificates."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];source='results/research/ripper/degree_prior.json'
s=json.loads((ROOT/source).read_text());assert s['status']=='validated'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'text.color':'#17243a','axes.labelcolor':'#273449','axes.edgecolor':'#b7c1cf','svg.hashsalt':'degree-prior-v1'})
fig,ax=plt.subplots(figsize=(11.8,7.1));values=[s['grouped'][0]['exact_graphs'],s['grouped'][1]['exact_graphs'],s['primary']['certified_integral_exact_graphs']]
ax.bar(range(3),values,color=['#8793a6','#a47d36','#347e72'],width=.62)
for index,value in enumerate(values):ax.text(index,value+.7,f'{value}/32',ha='center',weight='bold')
ax.axhline(29,color='#ba7e25',lw=1.3,ls='--');ax.text(.01,29.5,'Frozen LP target: at least 29 certified exact graphs',transform=ax.get_yaxis_transform(),color='#986819',fontsize=10)
ax.set(ylim=(0,36),xticks=range(3),xticklabels=['Original candidate support','Top-810 costs\nGraph size supplied','Degree-20 LP\nEvery cell degree supplied'],ylabel='Exactly recovered Sudoku graphs',yticks=[0,8,16,24,32])
ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Supplied degree information recovers all 32 graphs; cardinality alone recovers 7',x=.08,ha='left',fontsize=16.5,weight='bold')
ax.set_title('All 32 LP outputs have exact primal/dual certificates; no fractional, uncertified or unresolved result',loc='left',fontsize=11.2,pad=14)
fig.text(.08,.04,
    'Same 32 H1as candidate supports and frozen integer-quantized final log-LR costs. No new observations or acquisition replay.\n'
    'Top-810 ties use pair index. LP fixes each cell degree at 20. Degree and cardinality are explicit structural priors.\n'
    'No primal rounding; portable integer certificates prove optimality of the returned LP solutions, without a uniqueness claim.\n'
    'Top-810 loses 45 true edge-bank pairs and retains 45 extras. Original support retains every true edge but has 2,588 extras.\n'
    '7.529 total process/child CPU-s; 0.415 s monitored phase wall (different timer origins). All LP calls total 0.0115 solver CPU-s.\n'
    '888,000 historical queries represented; no new source rows, solver inputs or GPU work. BP AdamW outside this source-only study; not run.',
    fontsize=9,color='#526075',linespacing=1.5)
fig.subplots_adjust(left=.08,right=.97,bottom=.35,top=.80)
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/degree-prior.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(fig)
(ROOT/'assets/figures/degree-prior-provenance.json').write_text(json.dumps(dict(
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    input_sha256={source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},matplotlib=matplotlib.__version__,scope=s['scope']),indent=2)+'\n')
