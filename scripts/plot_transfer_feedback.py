"""H1bf finite transfer result with explicit donor-acquisition overhead."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];source='results/research/ripper/transfer_feedback.json';s=json.loads((ROOT/source).read_text());assert s['status']=='validated'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'text.color':'#17243a','axes.labelcolor':'#273449','axes.edgecolor':'#b7c1cf','svg.hashsalt':'transfer-feedback-v1'})
fig,axes=plt.subplots(1,2,figsize=(11.8,7.5),gridspec_kw={'width_ratios':[1,1]})
ax=axes[0];ax.bar([0,1],[29,30],color=['#738196','#347e72'],width=.55)
for x,y in enumerate([29,30]):ax.text(x,y+.8,f'{y}/32',ha='center',weight='bold')
ax.axhline(29,color='#6d7482',ls='--',lw=1);ax.set(ylim=(0,35),xticks=[0,1],xticklabels=['Same-bank prefix\nArchived','Transferred prefix'],ylabel='Certified integral exact graphs / 32',yticks=[0,8,16,24,32]);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
ax=axes[1];ax.bar([0,1],[720000,885928],color=['#738196','#347e72'],width=.55)
for x,y in enumerate([720000,885928]):ax.text(x,y+20000,f'{y:,}',ha='center',weight='bold')
ax.set(ylim=(0,1000000),xticks=[0,1],xticklabels=['Same-bank prefix','Transfer + donors'],ylabel='Standalone deployment label union',yticks=[0,250000,500000,750000,1000000],yticklabels=['0','250k','500k','750k','1M']);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Fixed donor feedback clears the target without recipient feedback',x=.08,ha='left',fontsize=15,weight='bold')
fig.text(.08,.83,'30/32 ≥ 29; two paired wins and one loss. Additional donor acquisition prevents a superiority claim.',fontsize=11)
fig.text(.08,.035,
    'One source-ID-only donor permutation; every donor first 300 IDs disjoint from recipient 900 IDs. Shared indexed topology.\n'
    'All 900 recipient masks saved before recipient labels/latents; immutable donor feedback and coquery counts only.\n'
    '32 complete-label replacement gates: every queried value changed, every mask unchanged. 32 new + 32 archived exact proofs.\n'
    'Transfer deployment = 720k recipient + 240k warmup − 74,072 overlap = 885,928 labels; donor cost is included.\n'
    'Actual new research labels 274,376; reused 445,624. Examined source/noise streams; donor coupling, no population claim.\n'
    '20.042 aggregate CPU-s / 3.624 wall-s; no GPU or puzzle-solver work. Degree 20 supplied; transfer top810 0/32 exact.\n'
    'BP AdamW outside this source-only study; not run.',fontsize=9,color='#526075',linespacing=1.5)
fig.subplots_adjust(left=.08,right=.97,bottom=.38,top=.77,wspace=.34)
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/transfer-feedback.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(fig)
(ROOT/'assets/figures/transfer-feedback-provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256={source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},matplotlib=matplotlib.__version__,scope=s['scope']),indent=2)+'\n')
