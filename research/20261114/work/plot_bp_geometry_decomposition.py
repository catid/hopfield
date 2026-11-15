"""Finite H2bm source geometry decomposition; retention is an invariant."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.hashsalt'] = 'h2bm-sealed-source-geometry'
d = Path(sys.argv[1])
a = json.loads((d / 'analysis.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
names = ['BP AdamW own certificate', 'BP + original LP certificate', 'Simple-path geometry']
values = [r['rejected_wrong'] for r in a['controls_in_bp_first_order']]
axes[0].barh(names, values, color=['#777777', '#a34c44', '#137c8b'])
axes[0].invert_yaxis()
for i, value in enumerate(values):
    axes[0].text(value + 6, i, f'{value}/510', va='center', fontsize=10)
axes[0].axvline(459, color='#b26e38', linestyle='--')
axes[0].text(450, -0.5, '90% target', ha='right', color='#8e5629', fontsize=9)
axes[0].set_xlim(0, 600)
axes[0].set_xlabel('Archived wrong BP outputs rejected')
axes[0].set_title('Geometry detects all source errors', loc='left', fontsize=11)
seeds = ['0', '1', '2']
invalid = [a['per_seed_partition'][s]['invalid_geometry'] for s in seeds]
longer = [a['per_seed_partition'][s]['valid_longer'] for s in seeds]
axes[1].bar(seeds, invalid, color='#137c8b', label='Invalid simple-path geometry')
axes[1].bar(seeds, longer, bottom=invalid, color='#a34c44', label='Valid but longer path: 0')
for i, n in enumerate(invalid):
    axes[1].text(i, n + 3, str(n), ha='center', fontsize=11)
axes[1].set_ylim(0, 215)
axes[1].set_xlabel('Fixed BP model seed')
axes[1].set_ylabel('Failed model-input boards')
axes[1].set_title('Complete 510-case failure partition', loc='left', fontsize=11)
axes[1].legend(loc='upper center', fontsize=9, frameon=False)
for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('All 510 archived BP errors violate simple-path geometry', x=.02, ha='left', fontsize=13)
fig.text(.02, .085, 'All 12,288 sealed H2bl source rows; all 510 failures and 48 fixed controls independently reconstructed.', fontsize=9)
fig.text(.02, .05, 'The 11,778 shortest paths have valid geometry by definition. Retention is a consistency check, not a second endpoint.', fontsize=9)
fig.text(.02, .015, 'Source analysis only: no fresh accuracy, screen-cost or cascade result. Simple-path geometry does not guarantee shortestness.', fontsize=9)
fig.tight_layout(rect=[0, .15, 1, .94])
fig.savefig(d / 'bp-geometry-decomposition.png', dpi=160, metadata={'Software': 'Matplotlib'})
fig.savefig(d / 'bp-geometry-decomposition.svg', metadata={'Date': None})
plt.close(fig)
