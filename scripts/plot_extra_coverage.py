"""H1bj graph recovery versus added observation cost; no refitting."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];source=ROOT/'results/research/ripper/extra_coverage/summary.json';s=json.loads(source.read_text())
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#26334b','text.color':'#26334b','axes.edgecolor':'#b4c0d0','savefig.facecolor':'white','svg.hashsalt':'hopfield-H1bj-extra-coverage-v1'})
fig,axes=plt.subplots(1,2,figsize=(14,8));fig.subplots_adjust(left=.07,right=.97,bottom=.35,top=.77,wspace=.3)
fig.suptitle('Seven extra cells recover all fixed recipient graphs',x=.07,y=.965,ha='left',fontsize=22,fontweight='bold')
fig.text(.07,.89,'32/32 exact in both arms; four aligned and seven permuted gains, with no paired losses.',fontsize=14)
for x,old,new,name,color in [(0,28,32,'Aligned','#328273'),(1,25,32,'Permuted\n(descriptive)','#92769e')]:
    axes[0].plot([x-.16,x+.16],[old,new],color=color,linewidth=2,zorder=2)
    axes[0].scatter([x-.16,x+.16],[old,new],color=[color,color],s=[100,140],zorder=3)
    for dx,v in [(-.16,old),(.16,new)]:axes[0].text(x+dx,v+.5,str(v),ha='center',weight='bold')
    axes[0].text(x-.16,23.1,'25 cells',ha='center',fontsize=10);axes[0].text(x+.16,23.1,'32 cells',ha='center',fontsize=10)
axes[0].plot([-.32,.32],[31,31],color='#bd6c69',linestyle='--');axes[0].text(-.32,30.4,'Aligned target: 31',color='#ad5b59',fontsize=10)
axes[0].set(xlim=(-.45,1.45),ylim=(22.5,34),xticks=[0,1],xticklabels=['Aligned','Permuted\n(descriptive)'],yticks=[24,26,28,30,32],ylabel='Certified integral exact graphs / 32');axes[0].grid(axis='y',alpha=.15);axes[0].set_title('Same banks, larger observation budget',loc='left',fontsize=13)
bottom=np.zeros(2)
for vals,label,color in [([720,720],'Original recipient labels','#328273'),([0,201.6],'Additional recipient labels','#c58e42'),([240,240],'Historical donor warmup','#b4becd')]:
    axes[1].bar([0,1],vals,bottom=bottom,color=color,width=.55,label=label)
    for i,v in enumerate(vals):
        if v:axes[1].text(i,bottom[i]+v/2,f'{v:g}k',ha='center',va='center',color='white' if color!='#b4becd' else '#26334b',weight='bold')
    bottom+=vals
for i,total in enumerate(bottom):axes[1].text(i,total+20,f'{total:,.1f}k',ha='center',weight='bold')
axes[1].set(xticks=[0,1],xticklabels=['25 cells / row','32 cells / row'],ylim=(0,1260),yticks=[0,400,800,1200],ylabel='Labels per complete alternative (thousands)');axes[1].set_title('Standalone deployment cost, each arm',loc='left',fontsize=13)
fig.legend(*axes[1].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.28),ncol=3,frameon=False,fontsize=10)
fig.text(.07,.22,'Aligned degree-selection and exact-graph targets both passed: 32/32 versus required 31, separately.\n'
         'Actual paired research increment: 154,425 labels after cross-arm reuse; each standalone arm adds 201,600.\n'
         'All 64 masks sealed before extra labels; no new identities or corruption draws. Original-budget H1bh failures retained.\n'
         '2,560 exact LP proofs; 7,680 mutations rejected; 64 complete-label gates. 30.173 CPU-s / 3.975 wall-s.\n'
         'BP AdamW outside this source-only study; not run. Extended top-810: 16/32 aligned, 12/32 permuted.\n'
         'Common-degree grid and historical donor feedback remain supplied. No new alignment or significance claim.',
         fontsize=10.5,color='#596a83',va='top',linespacing=1.5)
out=ROOT/'assets/figures'
for ext in ['png','svg']:fig.savefig(out/f'extra-coverage.{ext}',dpi=170,metadata={'Date':None})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(out/'extra-coverage-provenance.json').write_text(json.dumps(dict(source=str(source.relative_to(ROOT)),source_sha256=sha(source),script_sha256=sha(Path(__file__)),png_sha256=sha(out/'extra-coverage.png'),svg_sha256=sha(out/'extra-coverage.svg'),status='rendered_from_frozen_summary_without_refitting'),indent=2)+'\n')
