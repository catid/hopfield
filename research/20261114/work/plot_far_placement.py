"""H2bq fresh joint accuracy and separately measured complete answer cost."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.hashsalt'] = 'h2bp-far-placement-cascade'
d = Path(__file__).resolve().parents[3] / 'results/research/work/bp-first-far-placement'
a = json.loads((d / 'audit.json').read_text())
costs = a['cost_endpoint']['pairs']
fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
names = ['BP AdamW executed prefix', 'LP96 standalone', 'BP → geometry → LP96 subset']
values = [a['joint_success'][arm] for arm in ['BP_AdamW','LP96','cascade']]
axes[0].barh(names, values, color=['#777777', '#b16c37', '#137c8b'])
axes[0].invert_yaxis()
for i, v in enumerate(values):
    axes[0].text(1800, i, f'{v}/4096', va='center', color='white', fontsize=11)
axes[0].set_xlim(0, 4300)
axes[0].set_xlabel('Fresh mazes jointly solved, all three fixed models')
axes[0].set_title('Pair0 accuracy; 4,096 opposite-region mazes', loc='left', fontsize=10)
seconds_control = [r['LP96_seconds'] for r in costs]
seconds_candidate = [r['candidate_seconds'] for r in costs]
x = list(range(4))
axes[1].bar([i - .18 for i in x], seconds_control, width=.36, color='#b16c37', label='LP96 complete answer cost')
axes[1].bar([i + .18 for i in x], seconds_candidate, width=.36, color='#137c8b', label='BP-first cascade complete cost')
for i, r in enumerate(costs):
    axes[1].text(i, max(r['LP96_seconds'], r['candidate_seconds']) * 1.03, (f'{100*r["saving"]:.1f}%\nsaved' if r['saving']>=0 else f'{-100*r["saving"]:.1f}%\nhigher cost'), ha='center', fontsize=9)
axes[1].set_ylim(0, max(seconds_control + seconds_candidate) * 1.3)
axes[1].set_xticks(x, ['GPU0 pair0', 'GPU0 pair1', 'GPU1 pair2', 'GPU1 pair3'], rotation=15)
axes[1].set_ylabel('Measured complete answer seconds')
median=a['cost_endpoint']['median_saving']
headline=(f'Median saving {100*median:.2f}%; target ≥10%' if median>=0 else f'Cost target falsified: median {-100*median:.2f}% higher')
axes[1].set_title(headline,loc='left',fontsize=10)
axes[1].legend(loc='upper center', fontsize=8, frameon=False)
for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Geometry-routed BP-first cascade: opposite-region endpoint stress', x=.02, ha='left', fontsize=13)
absolute=a['primary_endpoints']['absolute_joint_success_999'];ni=a['primary_endpoints']['paired_LP96_noninferiority']
fig.text(.02,.09,f"Absolute ≥99.9%: {absolute['decision']}; one-sided 95% lower {absolute['one_sided95_success_lower_pp']:.4f}%. Paired noninferiority: {ni['decision']}.",fontsize=9)
fig.text(.02,.055,'Cost includes readiness, transfer, hidden/decoder/certificate/mapping. Archives and barriers excluded; BP is a prefix.',fontsize=9)
fig.text(.02, .02, 'Separate prespecified accuracy and cost claims; no simultaneous 95% claim. Geometry does not guarantee shortestness.', fontsize=9)
fig.tight_layout(rect=[0, .15, 1, .94])
fig.savefig(d / 'bp-first-far-placement.png', dpi=160, metadata={'Software': 'Matplotlib'})
fig.savefig(d / 'bp-first-far-placement.svg', metadata={'Date': None})
plt.close(fig)
