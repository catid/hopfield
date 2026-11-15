"""Separate fixed procedure and alignment endpoints for H1bc."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];source='results/research/ripper/soft_feedback.json'
s=json.loads((ROOT/source).read_text());assert s['status']=='validated'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'text.color':'#17243a','axes.labelcolor':'#273449','axes.edgecolor':'#b7c1cf','svg.hashsalt':'soft-feedback-v1'})
fig,ax=plt.subplots(figsize=(12,7.5));values=[s['grouped'][0]['exact_graphs'],s['grouped'][1]['exact_graphs'],s['grouped'][3]['certified_integral_exact'],s['grouped'][5]['certified_integral_exact']]
ax.bar(range(4),values,color=['#738196','#a47d36','#347e72','#987f9e'],width=.6)
for x,y in enumerate(values):ax.text(x,y+.7,f'{y}/32',ha='center',weight='bold')
ax.axhline(29,color='#6d7482',ls='--',lw=1);ax.text(.01,29.5,'Aligned procedure target: at least 29 exact',transform=ax.get_yaxis_transform(),fontsize=9.5)
ax.set(ylim=(0,36),xticks=range(4),xticklabels=['Calibrated adaptive\nArchived','Label-blind balanced\nArchived','Aligned soft feedback','Permuted soft feedback'],ylabel='Certified integral exact graphs / 32',yticks=[0,8,16,24,32])
ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Aligned soft feedback passes two separate finite targets',x=.08,ha='left',fontsize=17,weight='bold')
ax.set_title('Procedure: 31/32 ≥ 29. Alignment difference: 31 − 26 = +5 ≥ +4. No unresolved proofs.',loc='left',fontsize=11.7,pad=15)
fig.text(.08,.04,
    'Weights (8/9)^coqueries / (1 + past equalities), exponent1; control permutes cell coordinates of feedback only. No calibration.\n'
    'Each arm maintains its own real history; current mask finalized before current labels/latents. Same source rows, noise, and tie streams.\n'
    '64 new + 64 archived exact proofs, 64 future-suffix gates. All new solutions integral; six paired aligned wins and one loss.\n'
    'Different feedback trajectories include exposure/magnitude pathways; neither endpoint identifies a unique causal mediator.\n'
    'Each alternative deployment: 720k queries, zero calibration. Actual incremental union402,797; cumulative physical union2,021,949.\n'
    'Previously examined sources and latents; degree20 still supplied. BP AdamW outside this source-only study; not run.\n'
    '32.541 aggregate CPU-s / 5.040 wall-s including sequential replay, suffix checks and proofs; no GPU or puzzle-solver work.',fontsize=9,color='#526075',linespacing=1.5)
fig.subplots_adjust(left=.08,right=.97,bottom=.37,top=.80)
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/soft-feedback.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(fig)
(ROOT/'assets/figures/soft-feedback-provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    input_sha256={source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},matplotlib=matplotlib.__version__,scope=s['scope']),indent=2)+'\n')
