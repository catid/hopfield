"""H1bp separated recovery and advantage endpoints from frozen evidence."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];source=ROOT/'results/research/ripper/balanced_masks/summary.json';s=json.loads(source.read_text())
assert [g['success'] for g in s['grouped']]==[32,32]
variance=[sum(r['coquery_summary']['variance_numerator']/r['coquery_summary']['variance_denominator'] for r in s['rows'] if r['arm']==arm)/32 for arm in ['uniform','balanced']]
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#26334b','text.color':'#26334b','axes.edgecolor':'#b4c0d0','savefig.facecolor':'white','svg.hashsalt':'hopfield-H1bp-balanced-masks-v1'})
fig,axes=plt.subplots(1,2,figsize=(13.5,8));fig.subplots_adjust(left=.08,right=.97,bottom=.34,top=.76,wspace=.32)
fig.suptitle('Exact pair balance gives no recovery advantage at 180 rows',x=.08,y=.965,ha='left',fontsize=20,fontweight='bold')
fig.text(.08,.89,'Balanced recovery target supported: 32/32. Advantage target falsified: 0, required ≥2.',fontsize=13)
labels=['Uniform masks','Balanced masks'];colors=['#8497ae','#328273']
axes[0].bar(range(2),[32,32],color=colors,width=.6)
for x in range(2):axes[0].text(x,16,'32',ha='center',va='center',color='white',fontsize=18,weight='bold')
axes[0].plot([.65,1.35],[31,31],color='#bd6c69',linestyle='--');axes[0].text(1,33.1,'Target: 31',fontsize=10,color='#ad5b59',ha='center')
axes[0].set(xticks=range(2),xticklabels=labels,yticks=[0,8,16,24,32],ylim=(0,36),ylabel='Certified integral exact graphs /32');axes[0].set_title('All 64 arms choose degree 20',loc='left',fontsize=12)
axes[1].bar(range(2),variance,color=colors,width=.6)
axes[1].text(0,variance[0]/2,f'{variance[0]:.3f}',ha='center',va='center',color='white',fontsize=16,weight='bold');axes[1].text(1,.8,'0',ha='center',color=colors[1],fontsize=16,weight='bold')
axes[1].set(xticks=range(2),xticklabels=labels,yticks=[0,10,20,30],ylim=(0,35),ylabel='Mean within-bank pair-exposure variance');axes[1].set_title('Balanced: exactly 142 coqueries for every pair',loc='left',fontsize=12)
fig.text(.08,.23,'32 fresh paired banks; 180 boards ×72 cells; same latent iid 5% noise. Both deployments cost 414,720 labels.\n'
         'Randomized GF(9) affine-line complements balance pair exposure and jointly change mask dependencies.\n'
         'Physical noisy union =460,909; separate clean-source oracle =465,426. No unobserved labels used by inference.\n'
         '2,560 exact LP proofs; 7,680 altered proofs rejected; 64 full-label gates. BP AdamW outside scope; not run.\n'
         'All banks succeed under both masks. A ceiling result does not establish equivalence of the designs.',
         fontsize=10.5,color='#596a83',va='top',linespacing=1.6)
out=ROOT/'assets/figures'
for ext in ['png','svg']:fig.savefig(out/f'balanced_masks.{ext}',dpi=170,metadata={'Date':None})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(out/'balanced_masks-provenance.json').write_text(json.dumps(dict(source=str(source.relative_to(ROOT)),source_sha256=sha(source),script_sha256=sha(Path(__file__)),png_sha256=sha(out/'balanced_masks.png'),svg_sha256=sha(out/'balanced_masks.svg'),status='rendered_from_frozen_summary_without_refitting'),indent=2)+'\n')
