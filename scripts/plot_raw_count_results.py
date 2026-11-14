"""Render the raw-board study using only tracked, audited public evidence."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/figures'
sources = ['results/research/ripper/raw_count.json',
           'results/research/ripper/raw_compare.json']
structure, search = [json.loads((ROOT / p).read_text()) for p in sources]
assert structure['status'] == 'validated_exploratory_structure_stage'
assert search['status'] == 'validated'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'text.color': '#17243a', 'axes.labelcolor': '#273449',
    'xtick.color': '#273449', 'ytick.color': '#273449',
    'axes.edgecolor': '#b7c1cf', 'svg.hashsalt': 'hopfield-raw-count-v1'})
fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), gridspec_kw={'width_ratios': [1, 1.3]})
ns = np.array(structure['sizes'])
for seed in range(6):
    rows = [next(r for r in structure['rows'] if r['seed'] == seed and r['n'] == n) for n in ns]
    axes[0].plot(ns, [r['extra_edges'] for r in rows], marker='o', color='#16887d',
                 alpha=.45, lw=1, ms=4)
axes[0].set(xscale='log', yscale='symlog', xlabel='Raw complete training boards',
            ylabel='Extra edges beyond the 810 true peers', ylim=(-.2, 3200))
axes[0].set_xticks(ns, [str(n) for n in ns])
axes[0].set_yticks([0, 1, 10, 100, 1000], ['0', '1', '10', '100', '1,000'])
axes[0].set_title('Direct counts recover the graph', loc='left', weight='bold', pad=14)
axes[0].text(.04, .12, '100 boards: exact in all six banks\nNo true peer was excluded',
             transform=axes[0].transAxes, color='#116c63', fontsize=11, linespacing=1.5)
axes[0].grid(alpha=.16)
arms = [('pl_raw', 'PL: no augmentation', '#447bb7'),
        ('pl_digit', 'PL: digit relabeling', '#df8a3e'),
        ('count', 'Direct counts', '#16887d')]
x = np.arange(3)
for i, (arm, label, color) in enumerate(arms):
    rs = [next(r for r in search['grouped'] if r['n'] == n and r['arm'] == arm) for n in (30, 100, 300)]
    means = [r['mean_accuracy_pct'] for r in rs]
    xpos = x + (i-1)*.24
    axes[1].bar(xpos, means, width=.22, label=label, color=color, zorder=3)
    for pos, row in zip(xpos, rs):
        ys = 100*np.array(row['solved_per_seed'])/row['n_per_seed']
        axes[1].scatter(pos+np.linspace(-.055,.055,6), ys, s=14, color='#17243a',
                        alpha=.65, zorder=4, linewidth=0)
    for pos, mean in zip(xpos, means):
        if mean > 1:
            axes[1].text(pos, mean+3,
                         f'{mean:.1f}', ha='center', fontsize=10, color=color, weight='bold')
axes[1].set(xticks=x, xticklabels=['30', '100', '300'], xlabel='Raw complete training boards',
            ylabel='Solved test puzzles (%)', ylim=(-1, 100))
axes[1].set_title('Counts outperform these fixed PL recipes', loc='left', weight='bold', pad=14)
axes[1].legend(frameon=False, fontsize=10, loc='upper left', ncol=1)
axes[1].grid(axis='y', alpha=.16, zorder=0)
fig.suptitle('Constraint recovery without spatial augmentation', x=.065, ha='left',
             fontsize=21, weight='bold')
fig.text(.065, .025,
    'Six nested banks of raw, distinct official training solutions; no stored spatial variants. Dots show six seeds.\n'
    'Search: 10,000 fresh test puzzles, four chains, 262,144 moves per chain. PL: original 32-update recipes.\n'
    'Count N=300 reuses the byte-identical N=100 models and searches. Zero count wrong stops.\n'
    'The four registered count-versus-PL noninferiority tests passed the 5 pp margin; this is not a claim about all PL methods.',
    fontsize=9.5, color='#526075', linespacing=1.45)
fig.subplots_adjust(left=.065, right=.98, top=.80, bottom=.28, wspace=.32)
OUT.mkdir(parents=True, exist_ok=True)
for ext in ('png', 'svg'):
    fig.savefig(OUT / f'raw-count-pl.{ext}', dpi=190, bbox_inches='tight', facecolor='white', metadata={'Date':None} if ext=='svg' else None)
plt.close(fig)
(OUT / 'raw-count-provenance.json').write_text(json.dumps({
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'inputs_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
    'matplotlib': matplotlib.__version__, 'numpy': np.__version__,
    'scope': 'Public audited files only; no private checkpoints or artifact inputs.'}, indent=2)+'\n')
