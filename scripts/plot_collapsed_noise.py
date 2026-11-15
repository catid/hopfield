"""Finite-bank true-edge loss at matched nominal fixed-cell noise."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
source='results/research/ripper/collapsed_noise.json'
data=json.loads((ROOT/source).read_text());assert data['status']=='validated'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
    'text.color':'#17243a','axes.labelcolor':'#273449','xtick.color':'#273449','ytick.color':'#273449',
    'axes.edgecolor':'#b7c1cf','svg.hashsalt':'collapsed-noise-v1'})
fig,axes=plt.subplots(1,2,figsize=(13.8,7.4))
groups={group['arm']:group for group in data['grouped']}
labels=['Archived independent errors','Board-shared collapse'];colors=['#4976a8','#a44330']
for index,arm in enumerate(['iid','collapse']):
    group=groups[arm];values=np.array([row['missing_true_edges'] for row in data['rows'] if row['arm']==arm])
    axes[0].bar(index,values.mean(),width=.6,color=colors[index],alpha=.85,zorder=3)
    axes[0].scatter(index+np.linspace(-.15,.15,32),values,color='#17243a',s=16,alpha=.6,zorder=4)
    axes[0].text(index,values.max()+22,f'{group["banks_with_missing_true_edges"]}/32 banks lose edges',ha='center',weight='bold')
    errors=np.array([row['training_error_queries']/277.5 for row in data['rows'] if row['arm']==arm])
    axes[1].bar(index,group['training_error_pct'],width=.6,color=colors[index],alpha=.85,zorder=3)
    axes[1].scatter(index+np.linspace(-.15,.15,32),errors,color='#17243a',s=16,alpha=.6,zorder=4)
    axes[1].text(index,errors.max()+.25,f'Mean {group["training_error_pct"]:.3f}%',ha='center',weight='bold')
axes[0].set(ylim=(0,900),xticks=[0,1],xticklabels=labels,ylabel='Missing true edges per bank (of 810)')
axes[0].set_title('Structural failure: target at least 29/32 banks',loc='left',weight='bold',pad=14)
axes[0].grid(axis='y',alpha=.15)
axes[1].axhline(5,color='#ba7e25',ls='--',lw=1.5)
axes[1].text(.98,7.50,'Dashed: nominal fixed pre-query cell error is 5%',ha='right',color='#986819',transform=axes[1].get_yaxis_transform(),fontsize=9)
axes[1].set(ylim=(0,8),xticks=[0,1],xticklabels=labels,ylabel='Realized adaptive training-query error (%)')
axes[1].set_title('Measured errors on the adaptive query paths',loc='left',weight='bold',pad=14)
axes[1].grid(axis='y',alpha=.15)
fig.suptitle('Board-shared collapse destroys learned constraints at 5% marginal error',x=.065,ha='left',fontsize=19,weight='bold')
fig.text(.065,.035,
    '32 reused H1al source banks; M1000 / N1110; 29,750 queries per fit. Archived iid controls are reused, not rerun.\n'
    'Collapse event probability 9/160, followed by one uniform output digit for all queries/repeats on that board. Masks are fixed before the latent draw.\n'
    'For a fixed pre-query cell, error probability is (9/160)(8/9)=5%. Unequal clean-pair equality is 9/160; measured collapse true-pair equality is 5.717%.\n'
    'Collapse repeats always agree: calibrated upper bound 0.2647%, coverage 0/32. 25,302 of 25,920 true edge-bank pairs are lost.\n'
    f'No solver or invalid-certificate measurement. No new source/solver cohort or GPU work. {data["runtime"]["aggregate_cpu_s"]/60:.2f} CPU-min; {data["runtime"]["wall_s"]:.2f}s wall. BP AdamW outside scope.',
    color='#526075',fontsize=9.3,linespacing=1.5)
fig.subplots_adjust(left=.065,right=.98,bottom=.30,top=.82,wspace=.30)
for extension in ('png','svg'):
    fig.savefig(ROOT/f'assets/figures/collapsed-noise.{extension}',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(fig)
(ROOT/'assets/figures/collapsed-noise-provenance.json').write_text(json.dumps({
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'inputs_sha256':{source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},
    'matplotlib':matplotlib.__version__,'numpy':np.__version__,
    'scope':'H1am source-only finite-bank true-edge loss and realized adaptive-query errors, separated from nominal fixed-cell law.'},indent=2)+'\n')
