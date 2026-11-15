"""Plot the public H4aq exact consensus-objective diagnosis."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research/foureyes/consensus_gap'
summary = json.loads((BASE / 'summary.json').read_text())
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'svg.hashsalt': 'h4aq-consensus-gap-v1'})
fig, ax = plt.subplots(figsize=(11, 6))
labels = ['Cycle (degree 2)', 'Cycle square (degree 4)',
          'Cycle cube (degree 6)', 'Nine cliques (degree 8)']
strict = [g['strict_losses'] for g in summary['groups']]
tied = [g['tied_losses'] for g in summary['groups']]
ax.barh(range(4), strict, color='#b65b42', height=.58,
        label='True graph strictly more expensive')
ax.barh(range(4), tied, left=strict, color='#7d91a3', height=.58,
        label='True graph tied with optimum')
for i, (s, t) in enumerate(zip(strict, tied)):
    ax.text(s+t+.2, i, f'{s} strict + {t} tied', va='center')
ax.set_yticks(range(4), labels)
ax.invert_yaxis()
ax.set(xlim=(0, 14), xlabel='Previously exact graphs lost by bootstrap consensus')
ax.legend(loc='lower right', fontsize=10)
fig.suptitle('Bootstrap losses mostly come from changed preferences',
             x=.03, y=.97, ha='left', fontsize=17, weight='bold')
fig.text(.03, .88, '30 of 31 lost graphs have a strict vote-cost gap; the frozen ≥25 target passes.',
         fontsize=12)
fig.text(.03, .19,
         '29 losses have a cheaper wrong integral optimum; one has a cheaper fractional optimum.\n'
         'Choosing another optimum cannot repair those 30 losses. Only one lost graph remains tied.\n'
         'All 256 examined banks retained; no new observations, LP fits or solving results.\n'
         'Exact objective diagnosis; BP AdamW is outside this source-only study.',
         fontsize=10.5, linespacing=1.5, va='top', color='#405169')
fig.subplots_adjust(left=.24, right=.97, bottom=.31, top=.78)
for ext in ['png', 'svg']:
    fig.savefig(ROOT / f'assets/figures/consensus-gap.{ext}', dpi=160,
                metadata={'Date': None} if ext == 'svg' else {'Software': 'Matplotlib'})
plt.close(fig)
files = ['scripts/plot_consensus_gap.py', str((BASE/'summary.json').relative_to(ROOT)),
         str((BASE/'rows.json').relative_to(ROOT)), 'assets/figures/consensus-gap.png',
         'assets/figures/consensus-gap.svg']
(ROOT / 'assets/figures/consensus-gap-provenance.json').write_text(json.dumps(
    {'sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}},
    indent=2) + '\n')
