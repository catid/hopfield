"""Exposure balancing coverage effect and failed fixed recovery threshold."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];source='results/research/ripper/balanced_mask.json'
s=json.loads((ROOT/source).read_text());assert s['status']=='validated'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,
                     'text.color':'#17243a','axes.labelcolor':'#273449','axes.edgecolor':'#b7c1cf','svg.hashsalt':'balanced-mask-v1'})
fig,(left,right)=plt.subplots(1,2,figsize=(12.5,7.2),gridspec_kw={'width_ratios':[1.25,1]})
values=[32,s['grouped'][1]['certified_integral_exact'],s['primary']['certified_integral_exact_graphs']]
left.bar(range(3),values,color=['#738196','#a47d36','#347e72'],width=.6)
for x,y in enumerate(values):left.text(x,y+.7,f'{y}/32',ha='center',weight='bold')
left.axhline(29,color='#6d7482',ls='--',lw=1)
left.set(ylim=(0,36),xticks=range(3),xticklabels=['Adaptive\nArchived','Uniform\nArchived','Exposure\nbalanced'],ylabel='Certified integral exact graphs / 32',yticks=[0,8,16,24,32])
left.set_title('A   Target remains at least 29 exact graphs',loc='left',fontsize=11.5)
variance=[sum(row[arm]['dispersion'] for row in s['exposure_diagnostics'])/32/3240**2 for arm in ['uniform','balanced']]
right.bar(range(2),variance,color=['#a47d36','#347e72'],width=.55)
for x,y in enumerate(variance):right.text(x,y+1.4,f'{y:.3f}',ha='center',weight='bold')
right.set(ylim=(0,85),xticks=range(2),xticklabels=['Uniform','Exposure balanced'],ylabel='Mean within-bank pair-exposure variance')
right.set_title('B   Exposure variance falls by 98.505%',loc='left',fontsize=11.5)
for ax in [left,right]:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Balancing exposure alone does not reach the N900 recovery target',x=.07,ha='left',fontsize=17,weight='bold')
fig.text(.07,.04,
    'Primary falsified: 25/32 exact graphs and seven certified integral nonexact optima. No new fractional or unresolved runs.\n'
    'All masks fixed before labels; greedy weights (8/9)^coqueries, exclusions always false. Same tie streams, source rows and corrupted boards.\n'
    'Same unchanged Laplace-rate degree-20 LP; degree20 remains supplied. Balanced top-810: 0/32 exact, 250 missing and 250 extra pairs.\n'
    'Exposure variance is descriptive and does not identify the failure mechanism. Six uniform failures improve; four uniform successes worsen.\n'
    '720k-query zero-calibration alternative deployment; actual 337,225 new answers beyond both earlier masks, cumulative union 1,619,152.\n'
    '32 new + 64 archived exact proofs. BP AdamW outside this source-only study; not run. No fresh source IDs or latent draws.\n'
    '11.853 aggregate CPU-s / 2.367 wall-s including independent replay and proofs; zero puzzle-solver inputs or GPU work.',fontsize=9,color='#526075',linespacing=1.5)
fig.subplots_adjust(left=.07,right=.97,bottom=.36,top=.79,wspace=.32)
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/balanced-mask.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(fig)
(ROOT/'assets/figures/balanced-mask-provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    input_sha256={source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},matplotlib=matplotlib.__version__,scope=s['scope']),indent=2)+'\n')
