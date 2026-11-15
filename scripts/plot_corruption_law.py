"""H1bn separate failed iid endpoints and declared noise pair probabilities."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];source=ROOT/'results/research/ripper/corruption_law/summary.json';s=json.loads(source.read_text())
assert [g['success'] for g in s['grouped']]==[30,15,30]
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#26334b','text.color':'#26334b','axes.edgecolor':'#b4c0d0','savefig.facecolor':'white','svg.hashsalt':'hopfield-H1bn-corruption-law-v1'})
fig,axes=plt.subplots(1,2,figsize=(14,8));fig.subplots_adjust(left=.07,right=.97,bottom=.35,top=.77,wspace=.3)
fig.suptitle('Both iid noise targets fail; every arm still selects degree 20',x=.07,y=.965,ha='left',fontsize=21,fontweight='bold')
fig.text(.07,.89,'Required: 31/32 exact. Independent 5%: 15/32. Pair-matched independent: 30/32.',fontsize=14)
colors=['#7089a4','#bd6c69','#328273'];labels=['Bursty\n5% errors','IID\n5% errors','IID pair-matched\n2.279% errors']
axes[0].bar(range(3),[30,15,30],color=colors,width=.6)
for x,y in enumerate([30,15,30]):axes[0].text(x,y/2,str(y),ha='center',va='center',color='white',fontsize=18,weight='bold')
axes[0].plot([.65,2.35],[31,31],color='#bd6c69',linestyle='--');axes[0].text(1.5,32.2,'IID targets: 31',fontsize=10,color='#ad5b59',ha='center')
axes[0].set(xticks=range(3),xticklabels=labels,yticks=[0,8,16,24,32],ylim=(0,36),ylabel='Certified integral exact graphs / 32');axes[0].set_title('Bursty baseline is descriptive; no equivalence claim',loc='left',fontsize=11.5)
rates=[.5625,1.21484375,.5625];axes[1].bar(range(3),rates,color=colors,width=.6)
for x,y in enumerate(rates):axes[1].text(x,y/2,f'{y:.4f}%',ha='center',va='center',color='white',weight='bold')
axes[1].set(xticks=range(3),xticklabels=labels,yticks=[0,.5,1,1.5],ylim=(0,1.5),ylabel='Analytic true-edge equality probability (%)');axes[1].set_title('Equal marginal cell errors do not match pair noise',loc='left',fontsize=11.5)
fig.text(.07,.24,'32 fresh 124-row banks; common uniform 72-cell masks; independent corruption streams; no tuned noise parameter.\n'
         'IID 5% has 14 integral nonexact and 3 fractional failures. No unresolved banks; all 96 choose degree 20.\n'
         '3,840 exact LP proofs; 11,520 altered proofs rejected; 96 full-label gates. BP AdamW outside scope; not run.\n'
         'Each alternative costs 285,696 noisy labels; all three latent arms cost 857,088. Separate clean oracle: 321,003.\n'
         '3,963 new IDs outside the declared combined ledger only. Same finite targets; no difficult corruptions dropped.\n'
         'Matching true-edge equality does not match covariance, nonpeer rates or whole-board digit counts.',
         fontsize=10.5,color='#596a83',va='top',linespacing=1.5)
out=ROOT/'assets/figures'
for ext in ['png','svg']:fig.savefig(out/f'corruption_law.{ext}',dpi=170,metadata={'Date':None})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(out/'corruption_law-provenance.json').write_text(json.dumps(dict(source=str(source.relative_to(ROOT)),source_sha256=sha(source),script_sha256=sha(Path(__file__)),png_sha256=sha(out/'corruption_law.png'),svg_sha256=sha(out/'corruption_law.svg'),status='rendered_from_frozen_summary_without_refitting'),indent=2)+'\n')
