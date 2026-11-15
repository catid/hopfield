"""Nested prefix counts with every certified outcome category retained."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];source='results/research/ripper/prefix_rate.json'
s=json.loads((ROOT/source).read_text());assert s['status']=='validated'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'text.color':'#17243a','axes.labelcolor':'#273449','axes.edgecolor':'#b7c1cf','svg.hashsalt':'prefix-rate-v1'})
fig,ax=plt.subplots(figsize=(11.8,7.5));groups=[g for g in s['grouped'] if g['arm']=='degree20']
exact=[g.get('certified_integral_exact',g.get('exact_graphs',0)) for g in groups]
nonexact=[g.get('certified_integral_nonexact',0) for g in groups];fractional=[g.get('certified_fractional',0) for g in groups]
x=list(range(4));ax.bar(x,exact,color='#347e72',width=.58,label='Certified integral exact')
ax.bar(x,nonexact,bottom=exact,color='#c48b4e',width=.58,label='Certified integral nonexact')
ax.bar(x,fractional,bottom=[a+b for a,b in zip(exact,nonexact)],color='#9b819f',width=.58,label='Certified fractional')
for pos,value in enumerate(exact):ax.text(pos,33,f'{value}/32 exact',ha='center',weight='bold')
ax.axhline(29,color='#6d7482',ls='--',lw=1);ax.text(.02,29.5,'Primary N600 target: at least 29 exact',transform=ax.get_yaxis_transform(),fontsize=9.5)
ax.set(ylim=(0,36),xticks=x,xticklabels=['N100\n80k training queries','N300\n240k training queries','N600 (primary)\n480k training queries','N1110 (archived)\n888k training queries'],ylabel='Banks, out of 32',yticks=[0,8,16,24,32])
ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(loc='upper left',bbox_to_anchor=(0,1.20),ncols=3,frameon=False,fontsize=10)
fig.suptitle('600 observations are insufficient for reliable smoothed-rate graph recovery',x=.08,ha='left',fontsize=16,weight='bold')
fig.text(.08,.04,
    'Frozen primary falsified: 7/32 certified integral exact graphs at N600; all 25 other outcomes are resolved procedure failures.\n'
    'Degree20 supplied; all3,240 pairs eligible. Identical Laplace-rate costs at each literal prefix; no later observations enter fits.\n'
    '96 new optimality certificates,32 archived certificates,96 suffix-substitution gates. Fractional output does not exclude integral co-optima.\n'
    'Same-rate top810 exact:0/32 at N100,N300,N600;16/32 at N1110. BP AdamW outside this source-only study; not run.\n'
    'Nested historical budgets are alternatives,not additive spend; add64k historical calibration queries to each all-bank training total.\n'
    'Previously examined streams; no fresh-source or calibration-free-acquisition claim. Zero new queries/source rows/GPU work.\n'
    '5.256 aggregate CPU-s /1.137 wall-s, including counts,suffix gates and proofs. No unresolved runs.',fontsize=9,color='#526075',linespacing=1.5)
fig.subplots_adjust(left=.08,right=.97,bottom=.38,top=.76)
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/prefix-rate.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(fig)
(ROOT/'assets/figures/prefix-rate-provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    input_sha256={source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},matplotlib=matplotlib.__version__,scope=s['scope']),indent=2)+'\n')
