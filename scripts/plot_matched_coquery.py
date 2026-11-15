"""H1bm matched pair coverage: distinct recovery and advantage endpoints."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];source=ROOT/'results/research/ripper/matched_coquery/summary.json';s=json.loads(source.read_text())
assert [g['success'] for g in s['grouped']]==[30,31]
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#26334b','text.color':'#26334b','axes.edgecolor':'#b4c0d0','savefig.facecolor':'white','svg.hashsalt':'hopfield-H1bm-matched-coquery-v1'})
fig,axes=plt.subplots(1,2,figsize=(14,8));fig.subplots_adjust(left=.07,right=.97,bottom=.35,top=.77,wspace=.3)
fig.suptitle('Equal pair counts: 31/32 dense, 30/32 sparse',x=.07,y=.965,ha='left',fontsize=23,fontweight='bold')
fig.text(.07,.89,'Dense recovery target supported: 31 ≥ 31. Dense-advantage target falsified: 31 − 30 = 1 < +2.',fontsize=13.5)
colors=['#92769e','#328273'];axes[0].bar([0,1],[30,31],color=colors,width=.55)
for x,y in enumerate([30,31]):axes[0].text(x,y/2,str(y),ha='center',va='center',color='white',fontsize=18,weight='bold')
axes[0].plot([.73,1.27],[31,31],color='#bd6c69',linestyle='--');axes[0].text(.73,32.2,'Dense target: 31',fontsize=10,color='#ad5b59',ha='left')
labels=['32 cells × 639 boards','72 cells × 124 boards']
axes[0].set(xticks=[0,1],xticklabels=labels,yticks=[0,8,16,24,32],ylim=(0,36),ylabel='Certified integral exact graphs / 32');axes[0].set_title('2 dense wins; 1 sparse win; 29 shared successes',loc='left',fontsize=12)
axes[1].bar([0,1],[654.336,285.696],color=colors,width=.55)
for x,y in enumerate([654.336,285.696]):axes[1].text(x,y/2,f'{y:,.3f}k',ha='center',va='center',color='white',weight='bold')
axes[1].set(xticks=[0,1],xticklabels=labels,yticks=[0,200,400,600],ylim=(0,720),ylabel='Complete alternative labels (thousands)');axes[1].set_title('316,944 pair coqueries per bank in both arms',loc='left',fontsize=12)
fig.text(.07,.24,'20,313 fresh IDs; disjoint from the declared combined ledger only. All IDs and masks sealed before labels.\n'
         'Dense uses the fixed first 124 rows and contains sparse masks there; shared mask-independent corruption.\n'
         'All 64 select degree 20; no unresolved or selected fractional outputs. Top-810 exact: 0/32 sparse, 2/32 dense.\n'
         '2,560 exact LP proofs; 7,680 altered proofs rejected; 64 complete-label gates. New paired union: 813,056.\n'
         'BP AdamW outside this source-only study; not run. Per-pair exposure variance differs despite matched totals.\n'
         'Separate finite targets; no equivalence result or isolated causal explanation.',
         fontsize=10.5,color='#596a83',va='top',linespacing=1.5)
out=ROOT/'assets/figures'
for ext in ['png','svg']:fig.savefig(out/f'matched_coquery.{ext}',dpi=170,metadata={'Date':None})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(out/'matched_coquery-provenance.json').write_text(json.dumps(dict(source=str(source.relative_to(ROOT)),source_sha256=sha(source),script_sha256=sha(Path(__file__)),png_sha256=sha(out/'matched_coquery.png'),svg_sha256=sha(out/'matched_coquery.svg'),status='rendered_from_frozen_summary_without_refitting'),indent=2)+'\n')
