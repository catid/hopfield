"""H4z fixed-support penalty strength and prospective gain threshold."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'results/research/foureyes/weak_weight_study/summary.json'
PARENT = 'results/research/foureyes/canonical_confirm/summary.json'
s = json.loads((ROOT/SOURCE).read_text()); parent = json.loads((ROOT/PARENT).read_text())
assert s['status'] == 'validated'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'svg.hashsalt': 'weak-weight-study-v1', 'svg.fonttype': 'none'})
fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.8), gridspec_kw={'width_ratios': [1.5, 1]})
controls = {g['arm']: g for g in parent['grouped']}
values = [controls['adamw']['accuracy_pct']]+[g['accuracy_pct'] for g in s['grouped']]+[controls['restricted']['accuracy_pct']]
axes[0].bar(np.arange(5), values, color=['#7b8494', '#527eac', '#3e918b', '#236b65', '#b28d45'])
for i, value in enumerate(values):
    axes[0].text(i, value+1.4, f'{value:.2f}%', ha='center', fontsize=10)
axes[0].set(xticks=np.arange(5), xticklabels=['BP AdamW\narchived', 'Weak 1/16\narchived', 'Weak 1/64', 'Weak 1/256', 'Counts\narchived'],
            ylim=(0, 95), ylabel='Exact-match accuracy (%)', title='Same 64 banks and 5,000 seen inputs')
e = s['primary']; lo, hi = e['gain95_interval_pp']; gain = e['gain_pp']
axes[1].errorbar([gain], [0], xerr=[[gain-lo], [hi-gain]], fmt='o', color='#236b65', capsize=6, markersize=9)
axes[1].axvline(5, color='#b24a3b', linestyle='--', label='Frozen +5-point target')
axes[1].axvline(0, color='#87909a', linewidth=.8)
axes[1].text(gain, .22, f'+{gain:.3f} points\n95% CI [{lo:.3f}, {hi:.3f}]', ha='center', fontsize=11)
axes[1].set(xlim=(-.3, 5.7), ylim=(-.5, .65), yticks=[], xlabel='Weak 1/256 minus weak 1/16 (points)', title='Primary target: falsified')
axes[1].legend(loc='lower right', frameon=False, fontsize=10)
for ax in axes:
    ax.grid(axis='y' if ax is axes[0] else 'x', alpha=.15); ax.set_axisbelow(True)
fig.suptitle('Weaker positive penalties give only a small gain', x=.065, ha='left', fontsize=19, weight='bold')
fig.text(.065, .065, 'Strong weight 1; identical positive support and median-positive-coefficient/4 split. Four chains, 262,144 moves/chain.\n'
         'All three weight arms return correct answers only at the cap; none certifies. Positive reweighting preserves the zero set.\n'
         'Paired two-way bootstrap: 64 banks × 5,000 inputs, 5,000 draws. Post-hoc mechanism study; no fresh-generalization claim.\n'
         'BP AdamW and counts are separately fitted H4y controls. 128 new searches; four-worker search phase 150.039 seconds.',
         fontsize=10, color='#4d5967', linespacing=1.5)
fig.subplots_adjust(left=.065, right=.98, bottom=.32, top=.84, wspace=.28)
for ext in ['png', 'svg']:
    fig.savefig(ROOT/f'assets/figures/weak-weight-study.{ext}', dpi=180, facecolor='white', metadata={'Date': None} if ext == 'svg' else None)
receipt = {'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'input_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [SOURCE, PARENT]}}
(ROOT/'assets/figures/weak-weight-study-provenance.json').write_text(json.dumps(receipt, indent=2)+'\n')
