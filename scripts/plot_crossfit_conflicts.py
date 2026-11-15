"""Deterministic H4ax graph outcomes and paired regressions."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    source=ROOT/'results/research/foureyes/crossfit_conflicts/summary.json';s=json.loads(source.read_text());g=s['grouped'][0]
    plt.rcParams.update({'font.size':11,'svg.hashsalt':'h4ax-crossfit-v1'})
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    vals=[129,130,g['success']];bars=axes[0].bar(['Original counts\n600 rows','Same-row detector\n600 training rows','Opposite-fold\ndetector\n300 training rows'],vals,color=['#888888','#0072B2','#D55E00'])
    for b,v in zip(bars,vals):axes[0].text(b.get_x()+b.get_width()/2,v+4,str(v),ha='center')
    axes[0].axhline(162,color='black',linestyle='--',linewidth=1,label='Crossfit target: at least 162')
    axes[0].tick_params(axis='x',labelsize=9);axes[0].set_ylim(0,270);axes[0].set_ylabel('Exact graphs / 256 examined banks');axes[0].legend(frameon=False,fontsize=9)
    vals=[g['gains_vs_detector'],g['losses_vs_detector']];bars=axes[1].bar(['Gains','Losses'],vals,color=['#0072B2','#D55E00'])
    for b,v in zip(bars,vals):axes[1].text(b.get_x()+b.get_width()/2,v+2,str(v),ha='center')
    axes[1].axhline(8,color='black',linestyle='--',linewidth=1,label='Loss target: at most 8')
    axes[1].set_ylim(0,140);axes[1].set_ylabel('Paired changes vs same-row detector');axes[1].legend(frameon=False,fontsize=9)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Splitting detector training into disjoint halves hurts graph recovery')
    fig.text(.5,.055,'Both frozen targets fail. All 256 banks retained; 768 certified LPs, no unresolved fits.',ha='center',fontsize=10)
    fig.text(.5,.015,'Training sample size and row dependence both change; no unique causal mechanism is established.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.105,1,.94]);folder=ROOT/'assets/figures'
    for ext in ['png','svg']:fig.savefig(folder/f'crossfit-conflicts.{ext}',dpi=180,metadata={'Software':'Matplotlib'} if ext=='png' else {'Date':None})
    plt.close(fig)
    p=dict(input=str(source.relative_to(ROOT)),input_sha256=sha(source),renderer=str(Path(__file__).resolve().relative_to(ROOT)),renderer_sha256=sha(__file__),matplotlib=matplotlib.__version__,output_sha256={f'assets/figures/crossfit-conflicts.{e}':sha(folder/f'crossfit-conflicts.{e}') for e in ['png','svg']})
    (folder/'crossfit-conflicts-provenance.json').write_text(json.dumps(p,indent=2)+'\n')
if __name__=='__main__':main()
