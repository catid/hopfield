"""Two finite H1bd endpoints, with certified fractional output retained."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];source='results/research/ripper/fresh_soft_feedback.json'
s=json.loads((ROOT/source).read_text());assert s['status']=='validated'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'text.color':'#17243a','axes.labelcolor':'#273449','axes.edgecolor':'#b7c1cf','svg.hashsalt':'fresh-soft-feedback-v1'})
fig,ax=plt.subplots(figsize=(11.5,7.6));groups=[s['grouped'][1],s['grouped'][3]]
bottom=[0,0]
for key,label,color in [('certified_integral_exact','Certified integral exact','#347e72'),('certified_integral_nonexact','Certified integral nonexact','#ad8052'),('certified_fractional','Certified fractional','#987f9e')]:
    values=[g[key] for g in groups];ax.bar(range(2),values,bottom=bottom,color=color,width=.5,label=label)
    for x,value in enumerate(values):
        if value:ax.text(x,bottom[x]+value/2 if value>=2 else bottom[x]+value+.7,str(value),ha='center',va='center',weight='bold',color='white' if value>=2 else color)
    bottom=[x+y for x,y in zip(bottom,values)]
ax.axhline(29,color='#6d7482',ls='--',lw=1);ax.text(1.3,29.8,'Aligned target: 29',ha='left',va='center',fontsize=9.5)
ax.set(ylim=(0,35),xlim=(-.65,1.9),xticks=range(2),xticklabels=['Aligned soft feedback','Permuted soft feedback'],ylabel='Banks / 32',yticks=[0,8,16,24,32]);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
ax.legend(loc='upper left',bbox_to_anchor=(1.01,.93),frameon=False,fontsize=9)
fig.suptitle('Soft feedback confirms both targets on new ledger sources',x=.08,ha='left',fontsize=16,weight='bold')
ax.set_title('Procedure: 29/32 ≥ 29. Alignment difference: 29 − 20 = +9 ≥ +4. No unresolved proofs.',loc='left',fontsize=11,pad=15)
fig.text(.08,.035,
    '32 fresh banks × 900 rows; 28,595 new unique IDs, disjoint from the declared Ripper completion-source ledger only.\n'
    'Paired arms share new noise and ties; actual histories remain separate. Fixed exponent1, no calibration; degree20 supplied.\n'
    '64 exact proofs, 64 future-suffix gates; ten paired aligned wins and one loss. Fractional output retained as procedure failure.\n'
    'These finite targets do not establish a population bound or identify a unique feedback mediator.\n'
    'Each alternative: 720k queries. Actual paired new union1,216,156; shared labels223,844; calibration0.\n'
    '47.617 aggregate CPU-s / 20.724 wall-s including source audit, sequential replay and proofs. No GPU or puzzle-solver work.\n'
    'BP AdamW outside this source-only study; not run. Same-arm top810 controls: 0/32 exact for both.',fontsize=9,color='#526075',linespacing=1.5)
fig.subplots_adjust(left=.08,right=.76,bottom=.37,top=.80)
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/fresh-soft-feedback.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(fig)
(ROOT/'assets/figures/fresh-soft-feedback-provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    input_sha256={source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},matplotlib=matplotlib.__version__,scope=s['scope']),indent=2)+'\n')
