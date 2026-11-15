"""H1bg fixed-feedback contrast with fractional outcomes retained."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];source='results/research/ripper/permuted_transfer.json';s=json.loads((ROOT/source).read_text());assert s['status']=='validated'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'text.color':'#17243a','axes.labelcolor':'#273449','axes.edgecolor':'#b7c1cf','svg.hashsalt':'permuted-transfer-v1'})
fig,ax=plt.subplots(figsize=(11.6,7.3));groups=[s['grouped'][3],s['grouped'][1]];bottom=[0,0]
for key,label,color in [('certified_integral_exact','Certified integral exact','#347e72'),('certified_integral_nonexact','Certified integral nonexact','#ad8052'),('certified_fractional','Certified fractional','#987f9e')]:
    values=[g[key] for g in groups];ax.bar(range(2),values,bottom=bottom,color=color,width=.5,label=label)
    for x,value in enumerate(values):
        if value:ax.text(x,bottom[x]+value/2,str(value),ha='center',va='center',weight='bold',color='white')
    bottom=[a+b for a,b in zip(bottom,values)]
ax.set(ylim=(0,35),xlim=(-.6,1.65),xticks=[0,1],xticklabels=['Aligned donor feedback\nArchived','Cell-permuted donor feedback'],ylabel='Banks / 32',yticks=[0,8,16,24,32]);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(loc='upper left',bbox_to_anchor=(1.01,.94),frameon=False,fontsize=9)
fig.suptitle('Fixed donor-feedback alignment meets the +4 contrast target',x=.08,ha='left',fontsize=16,weight='bold')
ax.set_title('Archived 30 − prospective 26 = +4; six paired aligned wins and two losses. No unresolved proofs.',loc='left',fontsize=10.8,pad=16)
fig.text(.08,.035,
    'Same immutable donor matrices, one fixed cell permutation each; full off-diagonal feedback histograms exactly preserved.\n'
    'All recipient masks saved before labels; no recipient-label feedback. Coquery/feedback trajectory interactions remain.\n'
    'Known aligned comparator; only the finite contrast is tested. Shared examined sources and donor coupling; no population claim.\n'
    '32 new + 32 archived exact proofs; 32 complete-label gates. Two certified fractional outputs retained as procedure failures.\n'
    'Same 720k recipient + 240k donor budgets; exact deployment union 885,928 aligned / 885,892 control after overlap.\n'
    'Actual new labels 190,123; updated H1bd stream union 1,908,440. 20.603 CPU-s / 3.894 wall-s; no GPU/puzzle-solver work.\n'
    'Degree 20 supplied. BP AdamW outside this source-only study; not run. Both top810 controls: 0/32 exact.',fontsize=9,color='#526075',linespacing=1.5)
fig.subplots_adjust(left=.08,right=.76,bottom=.38,top=.80)
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/permuted-transfer.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(fig)
(ROOT/'assets/figures/permuted-transfer-provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256={source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},matplotlib=matplotlib.__version__,scope=s['scope']),indent=2)+'\n')
