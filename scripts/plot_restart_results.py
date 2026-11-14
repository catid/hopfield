"""Render the frozen restart comparison from public, independently audited data."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/research/foureyes/restart/summary.json'
OUT = ROOT / 'assets/figures'
data = json.loads(SOURCE.read_text())
assert data['status'] == 'validated'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'bold',
    'text.color': '#17243a', 'axes.labelcolor': '#273449',
    'axes.edgecolor': '#b7c1cf', 'svg.hashsalt': 'hopfield-restart-v1'})
fig, axes = plt.subplots(1, 2, figsize=(11, 5.3), gridspec_kw={'width_ratios': [1.2, 1]})
for arm, label, color, marker in [('no_restart', 'No restarts', '#447bb7', 'o'),
        ('restart524288', 'Restart every 524,288 moves', '#16887d', 's')]:
    rows = [r for r in data['grouped'] if r['arm'] == arm]
    axes[0].plot([r['cap'] for r in rows],
        [10000 - np.mean(r['per_seed_solved']) for r in rows],
        color=color, marker=marker, lw=2, label=label)
axes[0].set(xscale='log', yscale='log', xlabel='Moves per chain',
    ylabel='Unsolved puzzles per 10,000\n(mean across six seeds)')
axes[0].set_title('Nearly overlapping budget curves', loc='left', pad=13)
axes[0].legend(frameon=False, fontsize=9)
axes[0].grid(alpha=.17)
for y, name in enumerate(('primary', 'secondary')):
    row = data[name]
    value = row['gain_pp']
    lo, hi = row['paired95_interval_pp']
    axes[1].errorbar(value, y, xerr=[[value-lo], [hi-value]], fmt='o',
        color='#16887d' if y == 0 else '#8997a9', capsize=5, lw=2)
axes[1].axvline(0, color='#8997a9', lw=1)
axes[1].axvline(.1, color='#b75b4b', ls='--', lw=1.5, label='Primary target: +0.1 pp')
axes[1].set(yticks=[0, 1], yticklabels=['4,194,304\nprimary', '16,777,216\ndescriptive'],
    ylim=(1.6, -.6), xlim=(-.09, .13), xlabel='Restart accuracy gain (percentage points)')
axes[1].set_title('Paired 95% bootstrap intervals', loc='left', pad=13)
axes[1].legend(frameon=False, loc='lower left', fontsize=9)
axes[1].grid(axis='x', alpha=.17)
fig.suptitle('Fixed restarts miss the frozen improvement target',
    x=.07, ha='left', fontsize=20, weight='bold')
fig.text(.07, .025,
    '10,000 follow-up puzzles · four chains · six search seeds · zero wrong stops\n'
    'Same unit count energy and move caps; restart recomputations add work.\n'
    'The primary upper bound is +0.0933 pp, below the prespecified +0.1 pp target.',
    fontsize=10, color='#526075')
fig.subplots_adjust(left=.07, right=.98, top=.80, bottom=.26, wspace=.48)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / 'restart-results.png', dpi=190, bbox_inches='tight', facecolor='white')
fig.savefig(OUT / 'restart-results.svg', bbox_inches='tight', facecolor='white', metadata={'Date': None})
plt.close(fig)
(OUT / 'restart-provenance.json').write_text(json.dumps({
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'inputs_sha256': {str(SOURCE.relative_to(ROOT)): hashlib.sha256(SOURCE.read_bytes()).hexdigest()},
    'matplotlib': matplotlib.__version__, 'numpy': np.__version__}, indent=2) + '\n')
