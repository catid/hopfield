"""Specific smoothed-rate recovery without a supplied noise-law parameter."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];source='results/research/ripper/laplace_rate.json'
s=json.loads((ROOT/source).read_text());assert s['status']=='validated'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'text.color':'#17243a','axes.labelcolor':'#273449','axes.edgecolor':'#b7c1cf','svg.hashsalt':'laplace-rate-v1'})
fig,ax=plt.subplots(figsize=(11.8,7.3));values=[s['grouped'][i]['exact_graphs'] for i in range(3)]+[s['primary']['certified_integral_exact_graphs']]
ax.bar(range(4),values,color=['#8793a6','#738196','#a47d36','#347e72'],width=.6)
for x,y in enumerate(values):ax.text(x,y+.7,f'{y}/32',ha='center',weight='bold')
ax.axhline(29,color='#ba7e25',ls='--',lw=1.3);ax.text(.01,29.6,'Frozen new-LP target: at least 29 certified exact graphs',transform=ax.get_yaxis_transform(),color='#986819',fontsize=9.5)
ax.set(ylim=(0,36),xticks=range(4),xticklabels=['Original support','Exact-law degree-20 LP\nArchived comparator','Smoothed-rate top-810\nGraph size supplied','Smoothed-rate degree-20 LP\nEvery cell degree supplied'],ylabel='Exactly recovered Sudoku graphs',yticks=[0,8,16,24,32])
ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Smoothed equality rates recover all 32 graphs without supplied noise probabilities',x=.08,ha='left',fontsize=16.3,weight='bold')
ax.set_title('Fixed costs: round-even[10⁶ × (equalities + 1) / (coqueries + 2)]; degree 20 remains supplied',loc='left',fontsize=11.1,pad=14)
fig.text(.08,.04,
    'Same 32 H1ap observed streams; all pairs eligible. New rate-based LPs recover every graph with exact primal/dual certificates.\n'
    'Archived exact-law LP certificates reverified; those LPs are not rerun. Same-rate top-810 leaves 23 missing and 23 extra edge-bank pairs.\n'
    'This replaces likelihood/exposure weighting and adds Laplace smoothing; it does not isolate only the removal of a noise parameter.\n'
    'Specific fixed-bank success; degree/cardinality remain explicit priors. No uniqueness, fresh-source or universal correlated-noise claim.\n'
    '888,000 historical training queries plus 64,000 calibration queries; zero new observations, source rows, solver inputs or GPU work.\n'
    '2.654 aggregate CPU-s / 0.603 wall-s. BP AdamW is outside this source-only study; not run.',fontsize=9,color='#526075',linespacing=1.5)
fig.subplots_adjust(left=.08,right=.97,bottom=.35,top=.80)
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/laplace-rate.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(fig)
(ROOT/'assets/figures/laplace-rate-provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    input_sha256={source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},matplotlib=matplotlib.__version__,scope=s['scope']),indent=2)+'\n')
