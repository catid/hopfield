"""Prospective ledger-disjoint N900 graph recovery confirmation."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];source='results/research/ripper/fresh_rate.json'
s=json.loads((ROOT/source).read_text());assert s['status']=='validated'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'text.color':'#17243a','axes.labelcolor':'#273449','axes.edgecolor':'#b7c1cf','svg.hashsalt':'fresh-rate-v1'})
fig,ax=plt.subplots(figsize=(11.7,7.2));values=[s['grouped'][0]['exact_graphs'],s['grouped'][1]['exact_graphs'],s['primary']['certified_integral_exact_graphs']]
ax.bar(range(3),values,color=['#8793a6','#a47d36','#347e72'],width=.56)
for x,y in enumerate(values):ax.text(x,y+.7,f'{y}/32',ha='center',weight='bold')
ax.axhline(29,color='#ba7e25',ls='--',lw=1.3);ax.text(.01,29.6,'Frozen target: at least 29 certified exact graphs',transform=ax.get_yaxis_transform(),color='#986819',fontsize=9.5)
ax.set(ylim=(0,36),xticks=range(3),xticklabels=['Original acquisition support','Smoothed-rate top-810\nGraph size supplied','Smoothed-rate degree-20 LP\nEvery cell degree supplied'],ylabel='Exactly recovered Sudoku graphs',yticks=[0,8,16,24,32])
ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('N900 recovery confirms on sources outside the archived Ripper ledger',x=.08,ha='left',fontsize=16,weight='bold')
ax.set_title('32 new banks; fresh source identities, calibration, acquisition ties, and corruption draws',loc='left',fontsize=11.1,pad=14)
fig.text(.08,.04,
    '32/32 certified integral exact graphs. Top-810 leaves 66 missing and 66 extra edge-bank pairs; no unresolved LP outcomes.\n'
    '59,819 distinct selected source IDs are disjoint from the declared 1,928,477-ID Ripper ledger; this is not a global exposure claim.\n'
    'Identity manifest sealed before selected labels; independent CSV checks and all 28,800 predictable masks/noise states replayed.\n'
    '784,000 NEW queries: 720,000 training + 64,000 calibration. Degree 20 remains supplied; acquisition still uses calibration.\n'
    '32 exact optimality proofs; no uniqueness or universal-noise claim. BP AdamW outside this source-only study; not run.\n'
    '58.497 aggregate CPU-s / 29.408 wall-s including source preparation and audits. Zero puzzle-solver inputs or GPU work.',fontsize=9,color='#526075',linespacing=1.5)
fig.subplots_adjust(left=.08,right=.97,bottom=.35,top=.80)
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/fresh-rate.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(fig)
(ROOT/'assets/figures/fresh-rate-provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    input_sha256={source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},matplotlib=matplotlib.__version__,scope=s['scope']),indent=2)+'\n')
