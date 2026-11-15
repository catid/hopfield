"""Deterministic H1bv exact-recovery and paired-gain comparison."""
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
    source = ROOT/'results/research/ripper/fixed_270_confirm/summary.json'
    data = json.loads(source.read_text())
    plt.rcParams.update({'font.size': 10, 'svg.hashsalt': 'h1bv-fixed-270-confirm-v1'})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    exact = [row['success'] for row in data['grouped']]
    bars = axes[0].bar(['180-row prefix', '270 rows'], exact, color=['#999999', '#0072B2'], width=.58)
    for bar, value in zip(bars, exact):
        axes[0].text(bar.get_x()+bar.get_width()/2, value-8, f'{value}/64', ha='center', color='white', fontweight='bold')
    axes[0].axhline(61, color='#333333', linestyle='--', linewidth=1, label='270-row target ≥61')
    axes[0].set_ylim(0, 75)
    axes[0].set_ylabel('Exact graphs out of 64')
    axes[0].set_title('270-row recovery target supported')
    axes[0].legend(frameon=False, fontsize=9, loc='upper left')
    paired = data['paired_changes']
    gains, losses = len(paired['full270_wins']), len(paired['prefix180_wins'])
    bars = axes[1].bar(['Gains', 'Losses', 'Net gain'], [gains, losses, gains-losses], color=['#0072B2', '#999999', '#D55E00'], width=.58)
    for bar, value in zip(bars, [gains, losses, gains-losses]):
        axes[1].text(bar.get_x()+bar.get_width()/2, value+.25, str(value), ha='center')
    axes[1].axhline(8, color='#333333', linestyle='--', linewidth=1, label='Net-gain target ≥8')
    axes[1].set_ylim(0, 10)
    axes[1].set_ylabel('Paired banks')
    axes[1].set_title('Paired gain target falsified')
    axes[1].legend(frameon=False, fontsize=9, loc='upper left')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('H1bv: 64/64 exact at 270 rows, but only six additional recoveries', fontsize=12)
    fig.text(.5, .045, '64 fresh banks; 7.5% IID cell noise; 72 queried cells per row; all banks retained; no unresolved results.', ha='center', fontsize=9)
    fig.text(.5, .012, 'Physical query union: 1,244,160. The 180-row control reuses its literal prefix.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .09, 1, .92])
    folder = ROOT/'assets/figures'
    png, svg = [folder/f'fixed_270_confirm.{suffix}' for suffix in ['png', 'svg']]
    fig.savefig(png, dpi=180, metadata={'Software': 'Matplotlib'})
    fig.savefig(svg, metadata={'Date': None})
    plt.close(fig)
    provenance = dict(study='H1bv', input=str(source.relative_to(ROOT)), input_sha256=sha(source), script=str(Path(__file__).resolve().relative_to(ROOT)),
                      script_sha256=sha(__file__), matplotlib=matplotlib.__version__, numpy=np.__version__,
                      output_sha256={str(path.relative_to(ROOT)): sha(path) for path in [png, svg]})
    (folder/'fixed_270_confirm-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(provenance))


if __name__ == '__main__':
    main()
