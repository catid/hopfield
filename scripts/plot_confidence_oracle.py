"""Oracle intervention endpoints and certificate/capped-return decomposition."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'assets/figures'
source = 'results/research/ripper/confidence_oracle.json'
data = json.loads((ROOT/source).read_text())
assert data['status'] == 'validated'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False, 'text.color': '#17243a',
    'axes.labelcolor': '#273449', 'xtick.color': '#273449', 'ytick.color': '#273449',
    'axes.edgecolor': '#b7c1cf', 'svg.hashsalt': 'confidence-oracle-v1'})
fig, axes = plt.subplots(2, 2, figsize=(14, 10.4))
axes = axes.ravel()
labels = {'base': 'Archived\nevidence', 'repair_true': 'Restore weak\ntrue edges',
          'soften_strong_extra': 'Soften strong\nextra edges', 'delete_weak_extra': 'Remove weak\nextra edges'}
for x, group in enumerate(data['grouped']):
    values = np.array(group['solved_per_seed'])/50
    axes[0].bar(x, group['correct_certified_pct'], width=.65, color='#16887d',
                label='Certified' if x == 0 else None, zorder=3)
    axes[0].bar(x, group['correct_capped_pct'], bottom=group['correct_certified_pct'],
                width=.65, color='#d39a45', label='Capped return' if x == 0 else None, zorder=3)
    axes[0].scatter(x+np.linspace(-.16,.16,32), values, color='#17243a', alpha=.45, s=10, linewidth=0, zorder=4)
    axes[0].text(x, float(values.max())+2, f'{group["mean_accuracy_pct"]:.2f}%', ha='center', weight='bold')
axes[0].set(xticks=range(4), xticklabels=[labels[g['arm']] for g in data['grouped']],
            ylim=(0,105), ylabel='Canonical exact-match accuracy (%)')
axes[0].set_title('Three isolated oracle interventions', loc='left', weight='bold', pad=12)
axes[0].legend(loc='upper left', frameon=False, ncol=2, fontsize=9)
axes[0].grid(axis='y', alpha=.16)
titles = {'repair_true': 'Restore true edges: equivalence within ±5 pp',
          'soften_strong_extra': 'Soften strong extras: required +10 pp',
          'delete_weak_extra': 'Remove weak extras: required +10 pp'}
for ax, endpoint in zip(axes[1:], data['endpoints']):
    arm = endpoint['arm']
    lower, upper = endpoint['paired_two_way_ci95_pp']
    mean = endpoint['effect_pp']
    decision = endpoint['decision']
    color = '#16887d' if decision.startswith('supported') else '#a44330' if decision.startswith('falsified') else '#9a7f4d'
    verdict = 'Supported' if decision.startswith('supported') else 'Falsified' if decision.startswith('falsified') else 'Inconclusive'
    bounds = [-5,5] if arm == 'repair_true' else [10]
    if arm == 'repair_true':
        ax.axvspan(-5,5,color='#16887d',alpha=.07)
    for bound in bounds:
        ax.axvline(bound,ymin=.23,ymax=.66,color='#ba7e25',ls='--',lw=1.5)
    ax.plot([lower,upper],[.42,.42],color=color,lw=3)
    ax.scatter([mean],[.42],color=color,s=65,zorder=3)
    ax.plot([lower,lower],[.38,.46],color=color,lw=2)
    ax.plot([upper,upper],[.38,.46],color=color,lw=2)
    ax.text(.02,.85,verdict,transform=ax.transAxes,color=color,weight='bold',fontsize=15)
    ax.text(.02,.71,f'Mean {mean:.2f}; interval [{lower:.2f}, {upper:.2f}]',
            transform=ax.transAxes,color='#526075',fontsize=10)
    ax.set(xlim=(min(lower,min(bounds),0)-3,max(upper,max(bounds),0)+3),ylim=(0,1),yticks=[],
           xlabel='Oracle intervention minus archived evidence (pp)')
    ax.set_title(titles[arm],loc='left',weight='bold',pad=12)
    ax.grid(axis='x',alpha=.16)
fig.suptitle('Diagnosing the half-budget boundary with oracle edits',x=.06,ha='left',fontsize=21,weight='bold')
fig.text(.06,.025,
    'Post-hoc mechanism study on the same seen 32 H1ai banks and 5,000 puzzles. Oracle edits use known Sudoku geometry; not a deployable learner.\n'
    'Restore 82 weak true edges; soften 503 strong extras; remove 19,445 weak extras. No true edge is deleted. No new labels or test cohort.\n'
    'Repair and softening preserve the zero set; deletion expands it. All true edges remain positive, preserving zero-energy certificate soundness.\n'
    f'{data["wrong_certified_stops"]} invalid certificates; {data["certified_returns_in_unique_searches"]:,} certificates across unique searches. Three separate marginal 95% intervals; not selection adjusted.\n'
    f'{data["new_unique_searches"]} new searches; {data["archived_reuse_cells"]} archived reuse cells; {data["total_allocated_gpu_s"]/60:.2f} new GPU-min. Search settings stay fixed. BP AdamW was outside scope.',
    fontsize=9.2,color='#526075',linespacing=1.45)
fig.subplots_adjust(left=.065,right=.98,top=.89,bottom=.21,hspace=.50,wspace=.23)
for extension in ('png','svg'):
    fig.savefig(OUT/f'confidence-weight-oracle.{extension}',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(fig)
(OUT/'confidence-weight-oracle-provenance.json').write_text(json.dumps({
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'inputs_sha256': {source: hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},
    'matplotlib': matplotlib.__version__, 'numpy': np.__version__,
    'scope': 'H1aj post-hoc oracle mechanism diagnosis: three separate endpoints; certificate and capped-return decomposition.'},indent=2)+'\n')
