"""Deterministic H1bw equal-query graph recovery and observed cell errors."""
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
    sources = [ROOT/'results/research/ripper/repeat_query_tradeoff'/name for name in ['summary.json', 'noise-accounting.json']]
    data, noise = [json.loads(path.read_text()) for path in sources]
    plt.rcParams.update({'font.size': 10, 'svg.hashsalt': 'h1bw-repeat-query-tradeoff-v1'})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    labels = ['270 boards\n1 query per cell', '90 boards\n3 queries per cell']
    recovery = [row['success'] for row in data['grouped']]
    bars = axes[0].bar(labels, recovery, color=['#0072B2', '#D55E00'], width=.6)
    for bar, value in zip(bars, recovery):
        axes[0].text(bar.get_x()+bar.get_width()/2, value-8, f'{value}/64', ha='center', color='white', fontweight='bold')
    axes[0].axhline(61, color='#333333', linestyle='--', linewidth=1, label='Triple-query target ≥61')
    axes[0].set_ylim(0, 75)
    axes[0].set_ylabel('Exact graphs out of 64')
    axes[0].set_title('Same query budget; worse graph recovery')
    axes[0].legend(frameon=False, fontsize=9, loc='upper left')
    errors = [100*noise[name]['queried_error_rate'] for name in ['first', 'decoded']]
    bars = axes[1].bar(['Single-query labels', 'Decoded triple labels'], errors, color=['#0072B2', '#D55E00'], width=.6)
    for bar, value in zip(bars, errors):
        axes[1].text(bar.get_x()+bar.get_width()/2, value+.2, f'{value:.3f}%', ha='center')
    axes[1].set_ylim(0, 9)
    axes[1].set_ylabel('Observed queried-cell error (%)')
    axes[1].set_title('Decoding reduced label noise')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('H1bw: repeated-label denoising did not offset fewer independent boards', fontsize=12)
    fig.text(.5, .055, 'Each arm: 19,440 queried labels per bank. Triple-query arm: 14 paired losses (target ≤3), zero gains.', ha='center', fontsize=9)
    fig.text(.5, .018, '64 fresh paired banks; all retained; no unresolved results. Physical query union: 2,073,600.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .11, 1, .92])
    folder = ROOT/'assets/figures'
    png, svg = [folder/f'repeat_query_tradeoff.{suffix}' for suffix in ['png', 'svg']]
    fig.savefig(png, dpi=180, metadata={'Software': 'Matplotlib'})
    fig.savefig(svg, metadata={'Date': None})
    plt.close(fig)
    provenance = dict(study='H1bw', input_sha256={str(path.relative_to(ROOT)):sha(path) for path in sources},
                      script=str(Path(__file__).resolve().relative_to(ROOT)), script_sha256=sha(__file__),
                      matplotlib=matplotlib.__version__, numpy=np.__version__,
                      output_sha256={str(path.relative_to(ROOT)): sha(path) for path in [png, svg]})
    (folder/'repeat_query_tradeoff-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(provenance))


if __name__ == '__main__':
    main()
