"""Global zero-set proofs, explicit witnesses and unresolved graphs stay separate."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];source='results/research/ripper/permuted_soundness.json';certsource='results/research/ripper/permuted_soundness_certificates.json'
s=json.loads((ROOT/source).read_text());c=json.loads((ROOT/certsource).read_text());assert s['status']=='validated'
r=next(r for r in c['rows']if r['classification']=='unsound_witness');board=np.array(r['counterexample_search']['board']).reshape(9,9);pair=r['counterexample_search']['forced_equal_pair']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'text.color':'#17243a','axes.labelcolor':'#273449','axes.edgecolor':'#b7c1cf','svg.hashsalt':'permuted-soundness-v1'})
fig,axes=plt.subplots(1,2,figsize=(13,7.5),gridspec_kw={'width_ratios':[1.15,1]})
counts=[s['counts'][k]for k in ['sound_proof','unsound_witness','unresolved']]
axes[0].bar(range(3),counts,color=['#347e72','#a44330','#b38a43'],width=.65,zorder=3)
for i,n in enumerate(counts):axes[0].text(i,n+.7,str(n),ha='center',weight='bold')
axes[0].axhline(29,color='#ba7e25',ls='--',lw=1.4);axes[0].set(ylim=(0,33),ylabel='Source banks (all 32 retained)',xticks=range(3),xticklabels=['Sound proof','Invalid witness','Unresolved'])
axes[0].grid(axis='y',alpha=.15);axes[0].set_title('Frozen global-soundness target: inconclusive',loc='left',weight='bold',pad=14)
ax=axes[1];ax.set(xlim=(-.5,8.5),ylim=(8.5,-.5),xticks=[],yticks=[]);ax.set_aspect('equal')
for i in range(10):ax.axhline(i-.5,color='#546275',lw=1.8 if i%3==0 else .4);ax.axvline(i-.5,color='#546275',lw=1.8 if i%3==0 else .4)
for cell in pair:
 row,col=divmod(cell,9);ax.add_patch(plt.Rectangle((col-.48,row-.48),.96,.96,color='#f3d7d0',zorder=0))
for row in range(9):
 for col in range(9):ax.text(col,row,str(board[row,col]+1),ha='center',va='center',fontsize=14,color='#a44330'if row*9+col in pair else'#17243a',weight='bold'if row*9+col in pair else'normal')
ax.set_title(f'Explicit zero-energy invalid board: bank {r["bank"]}',loc='left',weight='bold',pad=14)
fig.suptitle('Structural edge loss mostly preserves soundness, but not always',x=.065,ha='left',fontsize=19,weight='bold')
fig.text(.065,.035,
    'All 32 H1ap learned supports unchanged; supplied Sudoku geometry is used only for this audit. No new source or puzzle rows.\n'
    '28 complete 27-unit proofs independently replayed; 2 explicit invalid colorings satisfy every retained inequality; 2 bounded searches unresolved.\n'
    'Support requires at least 29 proofs; falsification requires at least 4 explicit witnesses. Neither threshold is met. No timeout is called unsound.\n'
    f'Right: a global counterexample without clues, not a solver benchmark return. Highlighted peer cells share digit {board.ravel()[pair[0]]+1}, yet learned energy is exactly zero.\n'
    f'35 count-identity proof steps checked; 18 corrupted proofs rejected. {s["runtime"]["aggregate_cpu_s"]:.2f} CPU-s; {s["runtime"]["wall_s"]:.2f}s wall. BP AdamW outside scope; not run.',
    fontsize=9.2,color='#526075',linespacing=1.5)
fig.subplots_adjust(left=.065,right=.98,bottom=.30,top=.82,wspace=.30)
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/permuted-soundness.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(fig)
(ROOT/'assets/figures/permuted-soundness-provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()for p in [source,certsource]},matplotlib=matplotlib.__version__,numpy=np.__version__,scope=s['scope']),indent=2)+'\n')
