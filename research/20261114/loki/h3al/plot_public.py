"""Deterministic figure from tracked exact finite-population probabilities."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[4]
PUBLIC = ROOT/'results/research/loki/h3al'


def main():
    path = PUBLIC/'summary.json'
    s = json.loads(path.read_text())
    assert s['status'] == 'validated'
    plt.rcParams.update({'font.size':10.5, 'axes.spines.top':False, 'axes.spines.right':False, 'svg.fonttype':'none', 'svg.hashsalt':'cram-h3al-public'})
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5))
    fig.subplots_adjust(left=.075, right=.97, bottom=.35, top=.75, wspace=.31)
    fig.suptitle('64 positive acquisitions: exact per-fit reliability exceeds 90%', x=.065, y=.95, ha='left', fontsize=17, fontweight='bold')
    fig.text(.065, .878, 'SUPPORTED for each fixed BP AdamW fit · success means ≥95% exact N15 input capacity', fontsize=11.8, fontweight='bold')
    fig.text(.065, .819, 'Uniform sampling without replacement from 1,932 valid N13 boards; source-only witness deletion', fontsize=11)
    doses = [0, 1, 8, 32, 64]
    colors = ['#287f77', '#bd7144']
    for bank, color, label in [(0, colors[0], 'Banks 0 / 1 / 3'), (2, colors[1], 'Bank 2')]:
        rows = {r['dose']:r for r in s['rows'] if r['bank'] == bank}
        axes[0].plot(range(5), [rows[d]['success_probability_pct'] for d in doses], 'o-', color=color, lw=2, label=label)
    axes[0].axhline(90, color='#ae3f3a', ls='--', lw=1.5)
    axes[0].text(.07, 92, '90% per-fit target', color='#ae3f3a', fontsize=9)
    axes[0].axvspan(3.75, 4.25, color='#d7e6dc', alpha=.6, zorder=0)
    axes[0].set_xticks(range(5), ['0', '1', '8', '32', '64\nPRIMARY'])
    axes[0].set_xlim(-.15, 4.3)
    axes[0].set_ylim(-3, 105)
    axes[0].set_ylabel('P(exact capacity ≥95%) (%)')
    axes[0].set_title('A   Exact success probability by acquisition dose', loc='left', fontweight='bold', pad=14)
    axes[0].legend(frameon=False, loc='upper left', bbox_to_anchor=(.02,.77), fontsize=9)
    rows = [s['primary'][0], s['primary'][2]]
    for i, (r, color) in enumerate(zip(rows, colors)):
        axes[1].bar(i-.17, r['success_probability_pct'], width=.32, color=color)
        axes[1].bar(i+.17, r['exact_graph_probability_pct'], width=.32, color=color, alpha=.42)
        for x, key in [(i-.17, 'success_probability_pct'), (i+.17, 'exact_graph_probability_pct')]:
            axes[1].annotate(f"{r[key]:.2f}%", (x, r[key]), xytext=(0,5), textcoords='offset points', ha='center', fontsize=9)
    axes[1].set_xticks([0,1], ['Banks 0 / 1 / 3', 'Bank 2'])
    axes[1].set_ylim(0, 111)
    axes[1].set_ylabel('Exact probability at 64 acquisitions (%)')
    axes[1].set_title('B   Capacity success versus exact graph recovery', loc='left', fontweight='bold', pad=14)
    axes[1].text(.04, .97, 'Solid: ≥95% capacity   ·   Pale: exact graph', transform=axes[1].transAxes, fontsize=9, va='top')
    for ax in axes:
        ax.grid(axis='y', alpha=.17)
        ax.set_axisbelow(True)
    fig.text(.065, .255, 'Per-fit reliability: 97.23% / 97.23% / 95.67% / 97.23%. Independent-stream joint success: 87.94%.', fontsize=11, fontweight='bold')
    fig.text(.065, .19, 'The joint product is descriptive and is not the primary ≥90% endpoint. Lower doses and exact graph recovery are descriptive.\nFull within-board class co-occurrence is retained; exact integer masses are checked by two different algorithms.', fontsize=10.7, linespacing=1.5)
    fig.text(.065, .12, 'All 80 bank masks and 32 byte-distinct supports retained. Same seen 831,605-key N15 universe; no new target cohort.\nThe full source pool evaluates sampling reliability; each learner receives only its own 64 selected positives.', fontsize=10.7, linespacing=1.5)
    fig.text(.065, .05, f"Whole stage: {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s. Conditional finite-population result, not generalization.", fontsize=10.3, color='#444')
    for ext in ['png', 'svg']:
        fig.savefig(PUBLIC/f'acquisition-reliability.{ext}', dpi=180, facecolor='white', metadata={'Date':None} if ext == 'svg' else None)
    receipt = dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [path, Path(__file__)]}, output_sha256={f'acquisition-reliability.{ext}':hashlib.sha256((PUBLIC/f'acquisition-reliability.{ext}').read_bytes()).hexdigest() for ext in ['png', 'svg']})
    (PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(receipt['output_sha256'])


if __name__ == '__main__':
    main()
