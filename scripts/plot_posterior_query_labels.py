"""Deterministic H1bz posterior versus archived hard-label graph comparison."""
from pathlib import Path
import hashlib, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source = ROOT/'results/research/ripper/posterior_query_labels/summary.json'
    data = json.loads(source.read_text())
    plt.rcParams.update({'font.size': 10, 'svg.hashsalt': 'h1bz-posterior-query-labels-v1'})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    x = np.arange(2)
    baseline = [group['old_exact'] for group in data['grouped']]
    actual = [group['success'] for group in data['grouped']]
    for delta, values, color, label in [(-.18, baseline, '#999999', 'Archived H1bw'), (.18, actual, '#0072B2', 'Posterior expected equalities')]:
        bars = axes[0].bar(x+delta, values, width=.35, color=color, label=label)
        for bar, value in zip(bars, values):
            axes[0].text(bar.get_x()+bar.get_width()/2, max(1,value-7), str(value), ha='center', color='white', fontweight='bold')
    axes[0].set_xticks(x, ['Single270', 'Triple90'])
    axes[0].set_ylim(0, 80)
    axes[0].set_ylabel('Exact graphs out of 64')
    axes[0].set_title('Unchanged acquired observations')
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    changes = data['paired_changes']['triple90']
    counts = [len(changes['gains']), len(changes['losses'])]
    bars = axes[1].bar(['Gains', 'Regressions'], counts, color=['#0072B2', '#D55E00'], width=.58)
    for bar, value in zip(bars, counts):
        axes[1].text(bar.get_x()+bar.get_width()/2, value+.15, str(value), ha='center')
    axes[1].set_ylim(0, max(5,max(counts)+2))
    axes[1].set_ylabel('Triple90 paired banks')
    endpoints = data['endpoints']
    gain, loss = endpoints['triple_exact'], endpoints['triple_regression']
    axes[1].set_title(f'Triple90 exact ≥58: {gain["decision"]}\nRegressions ≤1: {loss["decision"]}', fontsize=10)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('H1bz: posterior equality scoring from all acquired noisy labels', fontsize=12)
    unknown = sum(group['unresolved'] for group in data['grouped'])
    fig.text(.5, .05, f'All 128 arms retained; {unknown} unresolved. Full degree grids rerun; exact rational posterior costs and certificates.', ha='center', fontsize=9)
    fig.text(.5, .015, 'No new observations. Parent physical query union remains 2,073,600; each deployment used 1,244,160.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .105, 1, .91])
    folder = ROOT/'assets/figures'
    png, svg = [folder/f'posterior_query_labels.{suffix}' for suffix in ['png', 'svg']]
    fig.savefig(png, dpi=180, metadata={'Software': 'Matplotlib'})
    fig.savefig(svg, metadata={'Date': None})
    plt.close(fig)
    provenance = dict(study='H1bz', input=str(source.relative_to(ROOT)), input_sha256=sha(source),
                      script=str(Path(__file__).resolve().relative_to(ROOT)), script_sha256=sha(__file__),
                      matplotlib=matplotlib.__version__, numpy=np.__version__,
                      output_sha256={str(path.relative_to(ROOT)):sha(path) for path in [png,svg]})
    (folder/'posterior_query_labels-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance))


if __name__=='__main__':
    main()
