"""Prespecified intermediate prefix recovery on the same archived streams."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];source='results/research/ripper/intermediate_rate.json'
s=json.loads((ROOT/source).read_text());assert s['status']=='validated'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'text.color':'#17243a','axes.labelcolor':'#273449','axes.edgecolor':'#b7c1cf','svg.hashsalt':'intermediate-rate-v1'})
fig,ax=plt.subplots(figsize=(12,7.5));groups=sorted([g for g in s['grouped'] if g['arm']=='degree20'],key=lambda g:g['rows'])
exact=[g.get('certified_integral_exact',g.get('exact_graphs',0)) for g in groups]
nonexact=[g.get('certified_integral_nonexact',0) for g in groups];fractional=[g.get('certified_fractional',0) for g in groups]
x=list(range(5));ax.bar(x,exact,color='#347e72',width=.58,label='Certified integral exact')
ax.bar(x,nonexact,bottom=exact,color='#c48b4e',width=.58,label='Certified integral nonexact')
ax.bar(x,fractional,bottom=[a+b for a,b in zip(exact,nonexact)],color='#9b819f',width=.58,label='Certified fractional')
for pos,value in enumerate(exact):ax.text(pos,33,f'{value}/32 exact',ha='center',weight='bold')
ax.axhline(29,color='#6d7482',ls='--',lw=1)
ax.set(ylim=(0,36),xticks=x,xticklabels=['N600\nArchived','N750','N900\nPrimary','N1000','N1110\nArchived'],ylabel='Banks, out of 32',yticks=[0,8,16,24,32])
ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(loc='upper left',bbox_to_anchor=(0,1.20),ncols=3,frameon=False,fontsize=10)
fig.suptitle('900 observations recover all 32 fixed graphs with exact certificates',x=.08,ha='left',fontsize=16.5,weight='bold')
fig.text(.08,.04,
    'Frozen primary supported: N900 recovers 32/32 exact graphs, exceeding the target of 29 (dashed line). No unresolved outcomes.\n'
    'Adaptive follow-up after N600 failed; checkpoints 750, 900, 1000 frozen before counts. Same streams, no fresh-source claim.\n'
    '96 new and 64 archived exact optimality proofs; 96 future-suffix gates. Degree 20 supplied; all 3,240 pairs eligible.\n'
    'Same-rate top-810 exact: 0/32 at N750, 3/32 at N900, 11/32 at N1000. BP AdamW outside this source-only study; not run.\n'
    'N900 uses 720k historical training queries + 64k calibration = 784k total, versus 952k at N1110 (17.6% fewer total).\n'
    'Nested query budgets are alternatives, not additive spending. Acquisition used calibration. Zero new queries or GPU work.\n'
    '6.483 aggregate CPU-s / 1.359 wall-s, including independent counts, suffix gates, and proofs; no minimum-budget claim.',fontsize=9,color='#526075',linespacing=1.5)
fig.subplots_adjust(left=.08,right=.97,bottom=.36,top=.76)
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/intermediate-rate.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(fig)
(ROOT/'assets/figures/intermediate-rate-provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    input_sha256={source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},matplotlib=matplotlib.__version__,scope=s['scope']),indent=2)+'\n')
