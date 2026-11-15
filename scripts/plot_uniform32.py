"""H1bk fresh paired exact recovery and honest donor acquisition cost."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];source=ROOT/'results/research/ripper/uniform32/summary.json';s=json.loads(source.read_text())
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#26334b','text.color':'#26334b','axes.edgecolor':'#b4c0d0','savefig.facecolor':'white','svg.hashsalt':'hopfield-H1bk-uniform32-v1'})
fig,axes=plt.subplots(1,2,figsize=(14,8));fig.subplots_adjust(left=.07,right=.97,bottom=.35,top=.77,wspace=.3)
fig.suptitle('Donor-free uniform acquisition recovers all fresh graphs',x=.07,y=.965,ha='left',fontsize=22,fontweight='bold')
fig.text(.07,.89,'Uniform target supported: 32/32 ≥ 31. Donor-benefit target falsified: 32 − 32 = 0 < +2.',fontsize=14)
colors=['#328273','#92769e'];axes[0].bar([0,1],[32,32],color=colors,width=.55)
for x in [0,1]:axes[0].text(x,16,'32',ha='center',va='center',color='white',fontsize=18,weight='bold')
axes[0].plot([-.27,.27],[31,31],color='#bd6c69',linestyle='--');axes[0].text(-.27,31.5,'Uniform target: 31',fontsize=10,color='#ad5b59',bbox={'facecolor':'white','alpha':.9,'edgecolor':'none','pad':1})
axes[0].set(xticks=[0,1],xticklabels=['Uniform 32\n(no donor)','Donor 25 + 7'],yticks=[0,8,16,24,32],ylim=(0,35),ylabel='Certified integral exact graphs / 32');axes[0].set_title('Paired fresh banks; no wins or losses',loc='left',fontsize=13)
axes[1].bar([0,1],[921.6,921.6],color=colors,width=.55)
axes[1].bar(1,240,bottom=921.6,color='#b4becd',width=.55)
for x in [0,1]:axes[1].text(x,460.8,'921.6k\nrecipient',ha='center',va='center',color='white',weight='bold')
axes[1].text(1,1041.6,'240k donor',ha='center',va='center',fontsize=11);axes[1].text(0,948,'921.6k',ha='center',weight='bold');axes[1].text(1,1188,'1,161.6k',ha='center',weight='bold')
axes[1].set(xticks=[0,1],xticklabels=['Uniform 32\n(no donor)','Donor 25 + 7'],yticks=[0,400,800,1200],ylim=(0,1280),ylabel='Labels per complete alternative (thousands)');axes[1].set_title('240,000 fewer labels for uniform acquisition',loc='left',fontsize=13)
fig.text(.07,.24,'28,514 fresh IDs; disjoint from the declared combined Ripper/H4ak ledger only. All IDs and masks sealed before labels.\n'
         'Same common-degree grid, Laplace rates and 900 rows with shared corruption per bank; degree 20 selected on all 64.\n'
         '2,560 exact LP proofs; 7,680 altered proofs rejected; 64 complete-label gates. No unresolved banks.\n'
         'Actual paired new recipient union: 1,357,547. Historical donor labels reused for research, charged to deployment.\n'
         '49.787 CPU-s / 14.690 wall-s. BP AdamW outside this source-only study; not run. Top-810 controls: 14/32 each.\n'
         'Separate finite targets; no equivalence, significance, or universal absence-of-donor-value claim.',
         fontsize=10.5,color='#596a83',va='top',linespacing=1.5)
out=ROOT/'assets/figures'
for ext in ['png','svg']:fig.savefig(out/f'uniform32.{ext}',dpi=170,metadata={'Date':None})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(out/'uniform32-provenance.json').write_text(json.dumps(dict(source=str(source.relative_to(ROOT)),source_sha256=sha(source),script_sha256=sha(Path(__file__)),png_sha256=sha(out/'uniform32.png'),svg_sha256=sha(out/'uniform32.svg'),status='rendered_from_frozen_summary_without_refitting'),indent=2)+'\n')
