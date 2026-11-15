"""Paired hard-exclusion repair with explicit degree/cardinality priors."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];source='results/research/ripper/exclusion_repair.json'
s=json.loads((ROOT/source).read_text());assert s['status']=='validated'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'text.color':'#17243a','axes.labelcolor':'#273449','axes.edgecolor':'#b7c1cf','svg.hashsalt':'exclusion-repair-v1'})
fig,ax=plt.subplots(figsize=(11.8,7.5));values=[s['grouped'][0]['exact_graphs'],s['grouped'][1]['exact_graphs'],s['grouped'][2]['certified_integral_exact'],s['grouped'][3]['certified_integral_exact']]
ax.bar(range(4),values,color=['#8793a6','#a47d36','#b65c49','#347e72'],width=.6)
for x,y in enumerate(values):ax.text(x,y+.7,f'{y}/32',ha='center',weight='bold')
ax.text(2,17,'21/32 restricted LPs\ncertifiably infeasible',ha='center',color='#a04c35',fontsize=10)
ax.axhline(29,color='#ba7e25',ls='--',lw=1.3);ax.text(.01,29.5,'Frozen unrestricted target: at least 29 certified exact graphs',transform=ax.get_yaxis_transform(),color='#986819',fontsize=10)
ax.set(ylim=(0,36),xticks=range(4),xticklabels=['Original support','Top-810\nAll pairs eligible','Degree-20 LP\nOriginal restrictions','Degree-20 LP\nAll pairs eligible'],ylabel='Exactly recovered Sudoku graphs',yticks=[0,8,16,24,32])
ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Removing premature candidate exclusions repairs all 32 graphs',x=.08,ha='left',fontsize=17,weight='bold')
ax.set_title('Same observed streams, exact-null costs and supplied degree prior; candidate availability changes',loc='left',fontsize=11.3,pad=14)
fig.text(.08,.04,
    'Original H1ap graphs lose 41 true edge-bank pairs in 21 banks. All 21 have a cell with fewer than 20 allowed neighbors.\n'
    '43 exact LP optimality proofs and 21 integer degree-deficit infeasibility proofs; no primal rounding or unresolved cases.\n'
    'Unrestricted degree-20 repair restores all 41 omitted true pairs and removes all 45 extras. Top-810 leaves 24 missing and 24 extra pairs.\n'
    'Degree 20, cardinality 810 and exact permutation-law q0 = 9/1600 are explicit supplied priors. No uniqueness or fresh-source claim.\n'
    '888,000 historical training queries plus 64,000 calibration queries; zero new observations, source rows, solver inputs or GPU work.\n'
    '2.582 aggregate CPU-s / 0.582 wall-s, including scientific preparation and audit. BP AdamW outside this source-only study; not run.',
    fontsize=9,color='#526075',linespacing=1.5)
fig.subplots_adjust(left=.08,right=.97,bottom=.35,top=.80)
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/exclusion-repair.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(fig)
(ROOT/'assets/figures/exclusion-repair-provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    input_sha256={source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},matplotlib=matplotlib.__version__,scope=s['scope']),indent=2)+'\n')
