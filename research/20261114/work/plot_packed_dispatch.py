"""H2bs actual six-pair offline packed/original complete-cost measurements."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.hashsalt']='h2bs-packed-dispatch'
d=Path(__file__).resolve().parents[3]/'results/research/work/bp-first-packed-dispatch';a=json.loads((d/'audit.json').read_text());endpoint=a['cost_endpoint'];rows=endpoint['pairs'];x=list(range(6))
fig,ax=plt.subplots(figsize=(11,5.7));ax.bar([i-.18 for i in x],[r['original_order_seconds'] for r in rows],width=.36,color='#9a6c38',label='New original-order dispatcher control');ax.bar([i+.18 for i in x],[r['packed_seconds'] for r in rows],width=.36,color='#624095',label='Stable high-first packed dispatcher')
for i,r in enumerate(rows):ax.text(i,max(r['original_order_seconds'],r['packed_seconds'])*1.025,f"{100*r['saving']:+.2f}%",ha='center',fontsize=10)
ax.set_ylim(0,max(max(r['original_order_seconds'],r['packed_seconds']) for r in rows)*1.22);ax.set_xticks(x,[f'GPU{int(i>=3)} pair{i}' for i in x]);ax.set_ylabel('Complete cohort answer seconds');ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,loc='upper center',ncols=2,fontsize=9)
fig.suptitle(f"Stable offline batch packing: {100*endpoint['median_saving']:+.2f}% median saving; ≥10% target {endpoint['decision']}",x=.04,ha='left',fontsize=13)
ax.set_title('All six fixed pairs; 4,096 examined source inputs × three fixed models per arm',loc='left',fontsize=10)
fig.text(.04,.12,'Complete cost charges fresh readiness, actual queue construction/packing, solver/certificate work and global scatter.',fontsize=9)
fig.text(.04,.08,'All 147,456 timing copies preserve audited source outputs. True partial batches; no padded inputs or new accuracy observations.',fontsize=9)
fig.text(.04,.04,'BP AdamW and LP96 retain their labeled H2br source context; their historical costs are not controls for this endpoint.',fontsize=9)
fig.text(.04,.005,'Entire cohort available offline; no online queue latency, fresh generalization or superiority-to-other-solvers claim.',fontsize=9)
fig.tight_layout(rect=[0,.16,1,.94]);fig.savefig(d/'bp-first-packed-dispatch.png',dpi=160,metadata={'Software':'Matplotlib'});fig.savefig(d/'bp-first-packed-dispatch.svg',metadata={'Date':None});plt.close(fig)
