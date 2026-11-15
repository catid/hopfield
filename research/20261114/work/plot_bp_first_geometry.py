"""H2bn finite source preservation and separately measured complete cost."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.hashsalt'] = 'h2bn-actual-geometry-cascade'
d = Path(__file__).resolve().parents[3] / 'results/research/work/bp-first-geometry-cascade'
a = json.loads((d / 'audit.json').read_text())
pairs = a['source_preservation']['pairs']
costs = a['cost_endpoint']['pairs']
assert len({r['candidate_joint'] for r in pairs}) == 1
fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
names = ['BP AdamW executed prefix', 'LP96 standalone', 'BP → geometry → LP96 subset']
values = [pairs[0]['BP_executed_prefix_joint'], 4095, pairs[0]['candidate_joint']]
axes[0].barh(names, values, color=['#777777', '#b16c37', '#137c8b'])
axes[0].invert_yaxis()
for i, v in enumerate(values):
    axes[0].text(1800, i, f'{v}/4096', va='center', color='white', fontsize=11)
axes[0].set_xlim(0, 4300)
axes[0].set_xlabel('Source mazes jointly solved, all three models')
axes[0].set_title('Executed source preservation; no fresh accuracy', loc='left', fontsize=10)
seconds_control = [r['LP96_seconds'] for r in costs]
seconds_candidate = [r['candidate_seconds'] for r in costs]
x = list(range(4))
axes[1].bar([i - .18 for i in x], seconds_control, width=.36, color='#b16c37', label='LP96 complete answer cost')
axes[1].bar([i + .18 for i in x], seconds_candidate, width=.36, color='#137c8b', label='BP-first cascade complete cost')
for i, r in enumerate(costs):
    axes[1].text(i, max(r['LP96_seconds'], r['candidate_seconds']) * 1.03, f'{100*r["saving"]:.1f}% saved', ha='center', fontsize=9)
axes[1].set_ylim(0, max(seconds_control + seconds_candidate) * 1.3)
axes[1].set_xticks(x, ['GPU0 pair0', 'GPU0 pair1', 'GPU1 pair2', 'GPU1 pair3'], rotation=15)
axes[1].set_ylabel('Measured complete answer seconds')
axes[1].set_title(f'Median saving {100*a["cost_endpoint"]["median_saving"]:.2f}%; target ≥10%', loc='left', fontsize=10)
axes[1].legend(loc='upper center', fontsize=8, frameon=False)
for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Actual geometry-routed BP-first cascade on the complete source cohort', x=.02, ha='left', fontsize=13)
fig.text(.02, .09, 'Each arm and pair: all 4,096 source mazes × three fixed models. Every routed LP subset is actually executed.', fontsize=9)
fig.text(.02, .055, 'Cost includes fresh model readiness, input/hidden work, decoders, geometry, certificates and output mapping.', fontsize=9)
fig.text(.02, .02, 'Scientific archives/audits and host barriers are excluded from answer cost. Simple-path geometry does not guarantee shortestness.', fontsize=9)
fig.tight_layout(rect=[0, .15, 1, .94])
fig.savefig(d / 'bp-first-geometry-cascade.png', dpi=160, metadata={'Software': 'Matplotlib'})
fig.savefig(d / 'bp-first-geometry-cascade.svg', metadata={'Date': None})
plt.close(fig)
