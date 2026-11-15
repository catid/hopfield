"""H1bl equal label budget and the cell-versus-board acquisition tradeoff."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];source=ROOT/'results/research/ripper/coverage_tradeoff/summary.json';s=json.loads(source.read_text())
assert [g['success'] for g in s['grouped']]==[6,32]
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#26334b','text.color':'#26334b','axes.edgecolor':'#b4c0d0','savefig.facecolor':'white','svg.hashsalt':'hopfield-H1bl-coverage-tradeoff-v1'})
fig,axes=plt.subplots(1,2,figsize=(14,8));fig.subplots_adjust(left=.07,right=.97,bottom=.35,top=.77,wspace=.3)
fig.suptitle('More cells per board improves recovery at equal label cost',x=.07,y=.965,ha='left',fontsize=21,fontweight='bold')
fig.text(.07,.89,'Dense target supported: 32/32 ≥ 31. Dense-advantage target supported: 32 − 6 = 26 ≥ +2.',fontsize=14)
colors=['#92769e','#328273'];axes[0].bar([0,1],[6,32],color=colors,width=.55)
for x,y in enumerate([6,32]):axes[0].text(x,y/2,str(y),ha='center',va='center',color='white',fontsize=18,weight='bold')
axes[0].plot([.73,1.27],[31,31],color='#bd6c69',linestyle='--');axes[0].text(.73,32.7,'Dense target: 31',fontsize=10,color='#ad5b59',ha='left')
labels=['32 cells × 450 boards','72 cells × 200 boards']
axes[0].set(xticks=[0,1],xticklabels=labels,yticks=[0,8,16,24,32],ylim=(0,36),ylabel='Certified integral exact graphs / 32');axes[0].set_title('26 dense wins; no sparse wins',loc='left',fontsize=13)
axes[1].bar([0,1],[223.2,511.2],color=colors,width=.55)
for x,y in enumerate([223.2,511.2]):axes[1].text(x,y/2,f'{y:,.1f}k',ha='center',va='center',color='white',weight='bold')
axes[1].set(xticks=[0,1],xticklabels=labels,yticks=[0,200,400,600],ylim=(0,600),ylabel='Pair coqueries per bank (thousands)');axes[1].set_title('Each arm costs 14,400 labels per bank',loc='left',fontsize=13)
fig.text(.07,.24,'14,328 fresh IDs; disjoint from the declared combined ledger only. All IDs and masks sealed before labels.\n'
         'Dense uses the fixed first 200 rows and contains sparse masks there; shared mask-independent corruption.\n'
         'All 64 select degree 20. Sparse: 22 integral nonexact + 4 fractional failures; no unresolved banks.\n'
         '2,560 exact LP proofs; 7,680 altered proofs rejected; 64 complete-label gates. Paired new union: 716,800.\n'
         'BP AdamW outside this source-only study; not run. Top-810 exact: 0/32 sparse, 25/32 dense.\n'
         'Fewer independent corrupted boards accompany more pair coqueries; no unique causal-mediation claim.',
         fontsize=10.5,color='#596a83',va='top',linespacing=1.5)
out=ROOT/'assets/figures'
for ext in ['png','svg']:fig.savefig(out/f'coverage_tradeoff.{ext}',dpi=170,metadata={'Date':None})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(out/'coverage_tradeoff-provenance.json').write_text(json.dumps(dict(source=str(source.relative_to(ROOT)),source_sha256=sha(source),script_sha256=sha(Path(__file__)),png_sha256=sha(out/'coverage_tradeoff.png'),svg_sha256=sha(out/'coverage_tradeoff.svg'),status='rendered_from_frozen_summary_without_refitting'),indent=2)+'\n')
