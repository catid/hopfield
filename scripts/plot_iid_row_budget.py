"""H1bo literal prefix/full comparison rendered from the frozen summary."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];source=ROOT/'results/research/ripper/iid_row_budget/summary.json';s=json.loads(source.read_text())
assert [g['success'] for g in s['grouped']]==[12,32]
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#26334b','text.color':'#26334b','axes.edgecolor':'#b4c0d0','savefig.facecolor':'white','svg.hashsalt':'hopfield-H1bo-iid-row-budget-v1'})
fig,axes=plt.subplots(1,2,figsize=(13.5,8));fig.subplots_adjust(left=.08,right=.97,bottom=.34,top=.76,wspace=.32)
fig.suptitle('Doubling rows rescues recovery under independent cell noise',x=.08,y=.965,ha='left',fontsize=20,fontweight='bold')
fig.text(.08,.89,'Full-bank target supported: 32/32 exact. The literal half-bank prefix recovers 12/32.',fontsize=13)
labels=['124-row prefix\nDescriptive','248-row full bank\nPrimary'];colors=['#8497ae','#328273']
axes[0].bar(range(2),[12,32],color=colors,width=.6)
for x,y in enumerate([12,32]):axes[0].text(x,y/2,str(y),ha='center',va='center',color='white',fontsize=18,weight='bold')
axes[0].plot([.65,1.35],[31,31],color='#bd6c69',linestyle='--');axes[0].text(1,33.1,'Target: 31',fontsize=10,color='#ad5b59',ha='center')
axes[0].set(xticks=range(2),xticklabels=labels,yticks=[0,8,16,24,32],ylim=(0,36),ylabel='Certified integral exact graphs /32');axes[0].set_title('All 64 arms select degree 20',loc='left',fontsize=12)
axes[1].bar(range(2),[285696,571392],color=colors,width=.6)
for x,y in enumerate([285696,571392]):axes[1].text(x,y/2,f'{y:,}',ha='center',va='center',color='white',fontsize=13,weight='bold')
axes[1].set(xticks=range(2),xticklabels=labels,yticks=[0,200000,400000,600000],ylim=(0,650000),ylabel='Standalone noisy-label deployment budget');axes[1].ticklabel_format(axis='y',style='plain');axes[1].set_title('Literal shared prefix; physical union =571,392',loc='left',fontsize=12)
fig.text(.08,.23,'32 fresh paired banks; 72 uniformly queried cells per board; common literal iid 5% noise stream and masks.\n'
         '20 full-bank wins, no prefix wins and 12 shared successes; paired contrast descriptive. No unresolved banks.\n'
         '2,560 exact LP proofs; 7,680 altered proofs rejected; 64 full-label gates. BP AdamW outside scope; not run.\n'
         '7,916 new source IDs outside the declared combined ledger; separate clean-source oracle =641,196 cells.\n'
         'Rows, noisy labels and pair coqueries all double. No unique mechanism or universal robustness claim.',
         fontsize=10.5,color='#596a83',va='top',linespacing=1.6)
out=ROOT/'assets/figures'
for ext in ['png','svg']:fig.savefig(out/f'iid_row_budget.{ext}',dpi=170,metadata={'Date':None})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(out/'iid_row_budget-provenance.json').write_text(json.dumps(dict(source=str(source.relative_to(ROOT)),source_sha256=sha(source),script_sha256=sha(Path(__file__)),png_sha256=sha(out/'iid_row_budget.png'),svg_sha256=sha(out/'iid_row_budget.svg'),status='rendered_from_frozen_summary_without_refitting'),indent=2)+'\n')
