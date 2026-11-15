"""Histogram conservation can hide corrupt labels from the balance filter."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];source='results/research/ripper/permuted_noise.json'
s=json.loads((ROOT/source).read_text());assert s['status']=='validated';g=s['grouped'][1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'text.color':'#17243a','axes.labelcolor':'#273449','axes.edgecolor':'#b7c1cf','svg.hashsalt':'permuted-noise-v1'})
fig,axes=plt.subplots(1,2,figsize=(13.6,7.3))
missing=np.array([r['missing_true_edges']for r in s['rows']]);axes[0].bar(np.arange(32),missing,color=np.where(missing>0,'#a44330','#347e72'),width=.76,zorder=3)
axes[0].set(xlabel='Source bank',ylabel='Missing true edges (of 810)',xticks=[0,7,15,23,31],ylim=(0,max(missing)+1))
axes[0].set_title('21/32 banks lose constraints; target at least 16',loc='left',weight='bold',pad=14)
labels=['Archived collapse + filter','Position permutation + filter']
for i,group in enumerate(s['grouped']):
    total=group['accepted_corrupt_boards']+group['rejected_boards'];accepted=group['accepted_corrupt_boards']/total*100
    axes[1].bar(i,accepted,width=.6,color=['#347e72','#a44330'][i],alpha=.85,zorder=3)
    axes[1].text(i,accepted+2,f'{group["accepted_corrupt_boards"]:,} / {total:,} accepted',ha='center',weight='bold')
axes[1].set(xticks=[0,1],xticklabels=labels,ylim=(0,112),ylabel='Corrupted training boards accepted (%)')
axes[1].set_title('A global digit-count check cannot detect this law',loc='left',weight='bold',pad=14)
for ax in axes:ax.grid(axis='y',alpha=.15)
fig.suptitle('Preserving digit counts lets shared corruption evade the filter',x=.065,ha='left',fontsize=19,weight='bold')
fig.text(.065,.035,
    '32 reused H1an source banks; fresh permutation-noise observations. M1000 / N1110; 29,750 queries per new fit. Archived collapse is not rerun.\n'
    'On event probability 9/160, uniformly permute all 81 cell positions. Every digit still occurs nine times, so every 25-label subset passes the balance filter.\n'
    'Fixed-cell error remains 5%; true-pair equality is (9/160)(8/80)=0.5625%. Shared repeated labels still agree, giving calibration coverage 0/32.\n'
    f'All {g["accepted_corrupt_boards"]:,} corrupt rows accepted; 41 true edge-bank pairs lost. Filtered and independently replayed unfiltered trajectories coincide.\n'
    f'No solver or invalid-certificate measurement. No new source rows or GPU work. {s["runtime"]["aggregate_cpu_s"]/60:.2f} CPU-min; {s["runtime"]["wall_s"]:.2f}s wall. BP AdamW outside scope.',
    fontsize=9.2,color='#526075',linespacing=1.5)
fig.subplots_adjust(left=.065,right=.98,bottom=.30,top=.82,wspace=.25)
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/permuted-noise.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(fig)
(ROOT/'assets/figures/permuted-noise-provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs_sha256={source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},matplotlib=matplotlib.__version__,numpy=np.__version__,scope=s['source_scope']),indent=2)+'\n')
