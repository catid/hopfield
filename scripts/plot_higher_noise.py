"""H1bq separated recovery and advantage endpoints from frozen evidence."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];source=ROOT/'results/research/ripper/higher_noise/summary.json';s=json.loads(source.read_text())
assert [g['success'] for g in s['grouped']]==[25,26]
variance=[sum(r['coquery_summary']['variance_numerator']/r['coquery_summary']['variance_denominator'] for r in s['rows'] if r['arm']==arm)/32 for arm in ['uniform','balanced']]
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#26334b','text.color':'#26334b','axes.edgecolor':'#b4c0d0','savefig.facecolor':'white','svg.hashsalt':'hopfield-H1bq-balanced-masks-v1'})
fig,axes=plt.subplots(1,2,figsize=(13.5,8));fig.subplots_adjust(left=.08,right=.97,bottom=.34,top=.76,wspace=.32)
fig.suptitle('Balanced masks miss the advantage target at 7.5% noise',x=.08,y=.965,ha='left',fontsize=20,fontweight='bold')
fig.text(.08,.89,'Recovery target supported: 26/32 ≥24. Advantage target falsified: +1, required ≥4.',fontsize=13)
labels=['Uniform masks','Balanced masks'];colors=['#8497ae','#328273']
axes[0].bar(range(2),[25,26],color=colors,width=.6)
for x,y in enumerate([25,26]):axes[0].text(x,y/2,str(y),ha='center',va='center',color='white',fontsize=18,weight='bold')
axes[0].plot([.65,1.35],[24,24],color='#bd6c69',linestyle='--');axes[0].text(1,27.1,'Target: 24',fontsize=10,color='#ad5b59',ha='center')
axes[0].set(xticks=range(2),xticklabels=labels,yticks=[0,8,16,24,32],ylim=(0,36),ylabel='Certified integral exact graphs /32');axes[0].set_title('All 64 arms choose degree 20',loc='left',fontsize=12)
axes[1].bar(range(2),variance,color=colors,width=.6)
axes[1].text(0,variance[0]/2,f'{variance[0]:.3f}',ha='center',va='center',color='white',fontsize=16,weight='bold');axes[1].text(1,.8,'0',ha='center',color=colors[1],fontsize=16,weight='bold')
axes[1].set(xticks=range(2),xticklabels=labels,yticks=[0,10,20,30],ylim=(0,35),ylabel='Mean within-bank pair-exposure variance');axes[1].set_title('Balanced: exactly 142 coqueries for every pair',loc='left',fontsize=12)
fig.text(.08,.23,'32 fresh paired banks; 180 boards ×72 cells; same latent iid 7.5% noise. Both deployments cost 414,720 labels.\n'
         'Randomized GF(9) affine-line complements balance pair exposure and jointly change mask dependencies.\n'
         'Physical noisy union =460,721; separate clean-source oracle =465,750. No unobserved labels used by inference.\n'
         '2,560 exact LP proofs; 7,680 altered proofs rejected; 64 full-label gates. BP AdamW outside scope; not run.\n'
         '4 balanced wins, 3 uniform wins, 22 shared successes and 3 shared failures. No equivalence claim.',
         fontsize=10.5,color='#596a83',va='top',linespacing=1.6)
out=ROOT/'assets/figures'
for ext in ['png','svg']:fig.savefig(out/f'higher_noise.{ext}',dpi=170,metadata={'Date':None})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(out/'higher_noise-provenance.json').write_text(json.dumps(dict(source=str(source.relative_to(ROOT)),source_sha256=sha(source),script_sha256=sha(Path(__file__)),png_sha256=sha(out/'higher_noise.png'),svg_sha256=sha(out/'higher_noise.svg'),status='rendered_from_frozen_summary_without_refitting'),indent=2)+'\n')
