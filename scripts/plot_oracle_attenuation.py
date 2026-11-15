"""Measured H4as graph recovery, including both failed targets."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/research/foureyes/oracle_attenuation'
s = json.loads((BASE/'summary.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
                     'axes.spines.top':False, 'axes.spines.right':False,
                     'svg.hashsalt':'h4as-oracle-attenuation-v1'})
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(4)
base = [r['baseline_exact'] for r in s['grouped']]
new = [r['success'] for r in s['grouped']]
for offset, values, color, label in [(-.18, base, '#507b98', 'Original count LP'),
                                     (.18, new, '#bb684c', 'Privileged-mask corrected LP')]:
    ax.bar(x+offset, values, width=.34, color=color, label=label)
    for i, value in enumerate(values):
        ax.text(i+offset, value+1, str(value), ha='center', weight='bold')
ax.set_xticks(x, ['Cycle\ndegree 2', 'Cycle square\ndegree 4',
                  'Cycle cube\ndegree 6', 'Nine cliques\ndegree 8'])
ax.set(ylim=(0, 69), ylabel='Exact graphs / 64 examined banks')
ax.legend(loc='upper right')
fig.suptitle('Knowing the noise locations does not rescue this correction',
             x=.03, y=.97, ha='left', fontsize=17, weight='bold')
fig.text(.03, .88, '129 → 88 exact graphs; one gain and 42 losses. Both frozen targets are falsified.', fontsize=12)
fig.text(.03, .19,
         'Required: ≥145/256 exact graphs and ≤8 lost baseline successes, as separate endpoints.\n'
         'All 256 banks retained: 88 exact, 154 integral nonexact, 14 fractional, zero unresolved.\n'
         'Same noisy counts and inferred degree; 1,382,400 privileged corruption-location indices added.\n'
         'All 512 baseline/new certificates checked. BP AdamW is outside this source-only study.',
         fontsize=10.5, linespacing=1.5, va='top', color='#405169')
fig.subplots_adjust(left=.08, right=.97, bottom=.33, top=.78)
for ext in ['png', 'svg']:
    fig.savefig(ROOT/f'assets/figures/oracle-attenuation.{ext}', dpi=160,
                metadata={'Date':None} if ext=='svg' else {'Software':'Matplotlib'})
plt.close(fig)
paths = ['scripts/plot_oracle_attenuation.py', str((BASE/'summary.json').relative_to(ROOT)),
         'assets/figures/oracle-attenuation.png', 'assets/figures/oracle-attenuation.svg']
(ROOT/'assets/figures/oracle-attenuation-provenance.json').write_text(json.dumps(
    {'sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}}, indent=2)+'\n')
