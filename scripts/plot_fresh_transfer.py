"""H1bh negative fresh-recipient confirmation, with separate fixed targets."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];source='results/research/ripper/fresh_transfer.json';s=json.loads((ROOT/source).read_text());assert s['status']=='validated'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'text.color':'#17243a','axes.labelcolor':'#273449','axes.edgecolor':'#b7c1cf','svg.hashsalt':'fresh-transfer-v1'})
fig,ax=plt.subplots(figsize=(11.7,7.5));groups=[s['grouped'][1],s['grouped'][3]];bottom=[0,0]
for key,label,color in [('certified_integral_exact','Certified integral exact','#347e72'),('certified_integral_nonexact','Certified integral nonexact','#ad8052'),('certified_fractional','Certified fractional','#987f9e')]:
    values=[g[key] for g in groups];ax.bar(range(2),values,bottom=bottom,color=color,width=.5,label=label)
    for x,value in enumerate(values):
        if value:ax.text(x,bottom[x]+value/2,str(value),ha='center',va='center',weight='bold',color='white')
    bottom=[a+b for a,b in zip(bottom,values)]
ax.axhline(29,color='#a95555',ls='--',lw=1);ax.text(1.3,29.8,'Aligned target: 29',ha='left',fontsize=9.5,color='#a95555')
ax.set(ylim=(0,35),xlim=(-.65,1.9),xticks=[0,1],xticklabels=['Aligned fixed feedback','Permuted fixed feedback'],ylabel='Fresh recipient banks / 32',yticks=[0,8,16,24,32]);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(loc='upper left',bbox_to_anchor=(1.01,.93),frameon=False,fontsize=9)
fig.suptitle('Fresh recipients fail both prespecified transfer targets',x=.08,ha='left',fontsize=16,weight='bold')
ax.set_title('Procedure: 28/32 < 29. Alignment: 28 − 25 = +3 < +4. Both falsified; no unresolved proofs.',loc='left',fontsize=11,pad=15)
fig.text(.08,.035,
    '32 new 900-row banks; 28,580 unique IDs, disjoint from the declared Ripper source ledger only. Historical donors fixed.\n'
    'All IDs sealed, then all 64 mask streams saved, then selected CSV labels materialized. No recipient-label feedback.\n'
    '64 exact proofs and 64 complete-label gates; seven paired aligned wins and four losses. Two fractional outputs retained.\n'
    'Earlier examined-recipient success is not pooled into these decisions. No donor/permutation/cutoff rescue or population claim.\n'
    'Each alternative = 720k recipients + 240k historical donor warmup = 960k. Actual new paired labels 1,217,792.\n'
    '49.514 aggregate CPU-s / 21.213 wall-s including source audit, replay and proofs. No GPU or puzzle-solver work.\n'
    'Degree 20 supplied. BP AdamW outside this source-only study; not run. Both top810 controls: 0/32 exact.',fontsize=9,color='#526075',linespacing=1.5)
fig.subplots_adjust(left=.08,right=.76,bottom=.38,top=.80)
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/fresh-transfer.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(fig)
(ROOT/'assets/figures/fresh-transfer-provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256={source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},matplotlib=matplotlib.__version__,scope=s['scope']),indent=2)+'\n')
