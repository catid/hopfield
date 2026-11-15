"""H1be success target and paired changes from frozen policy feedback."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];source='results/research/ripper/frozen_feedback.json';s=json.loads((ROOT/source).read_text());assert s['status']=='validated'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'text.color':'#17243a','axes.labelcolor':'#273449','axes.edgecolor':'#b7c1cf','svg.hashsalt':'frozen-feedback-v1'})
fig,axes=plt.subplots(1,2,figsize=(11.8,7.3),gridspec_kw={'width_ratios':[1.2,1]})
ax=axes[0];ax.bar([0,1],[29,29],color=['#738196','#347e72'],width=.55)
for x in [0,1]:ax.text(x,30,'29/32',ha='center',weight='bold')
ax.axhline(29,color='#6d7482',ls='--',lw=1);ax.set(ylim=(0,35),xticks=[0,1],xticklabels=['Full feedback\nArchived','Frozen after 300'],ylabel='Certified integral exact graphs / 32',yticks=[0,8,16,24,32]);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
ax=axes[1];ax.bar([0,1,2],[26,3,3],color=['#738196','#347e72','#ad8052'],width=.55)
for x,y in enumerate([26,3,3]):ax.text(x,y+.7,str(y),ha='center',weight='bold')
ax.set(ylim=(0,35),xticks=[0,1,2],xticklabels=['Both exact','Frozen wins','Frozen losses'],ylabel='Paired banks',yticks=[0,8,16,24,32]);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('First 300 feedback meets the fixed 900-row recovery target',x=.08,ha='left',fontsize=16,weight='bold')
fig.text(.08,.83,'Both arms 29/32, but different failure banks: three paired wins and three losses. No unresolved proofs.',fontsize=11)
fig.text(.08,.035,
    'Only policy equality feedback freezes. Actual coquery/equality counts continue, and recovery uses all 900 observed rows.\n'
    '32 new + 32 archived exact proofs, 32 future-suffix gates. Same examined H1bd sources, noise and tie streams.\n'
    'One fixed cutoff; no minimum-prefix, population-equivalence or universal-feedback-necessity claim. Degree 20 still supplied.\n'
    'Alternative deployment 720k queries; no query saving. New 227,785 labels / reused 492,215; updated H1bd stream union 1,443,941.\n'
    '19.394 aggregate CPU-s / 3.266 wall-s including replay and proofs. No GPU or puzzle-solver work.\n'
    'BP AdamW outside this source-only study; not run. Frozen-arm top810 control: 0/32 exact.',fontsize=9,color='#526075',linespacing=1.5)
fig.subplots_adjust(left=.08,right=.97,bottom=.36,top=.77,wspace=.34)
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/frozen-feedback.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(fig)
(ROOT/'assets/figures/frozen-feedback-provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256={source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},matplotlib=matplotlib.__version__,scope=s['scope']),indent=2)+'\n')
