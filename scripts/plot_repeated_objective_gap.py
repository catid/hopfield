"""Deterministic H1bx strict-exclusion versus cooptimal-failure diagnostic."""
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
    source = ROOT/'results/research/ripper/repeated_objective_gap/summary.json'
    data = json.loads(source.read_text())
    plt.rcParams.update({'font.size': 10, 'svg.hashsalt': 'h1bx-repeated-objective-gap-v1'})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    labels = ['Single270', 'Triple90']
    exact = [row['exact'] for row in data['grouped']]
    tied = [row['cooptimal_nonexact'] for row in data['grouped']]
    strict = [row['strict'] for row in data['grouped']]
    axes[0].bar(labels, exact, color='#0072B2', label='Exact output')
    axes[0].bar(labels, tied, bottom=exact, color='#E69F00', label='Cooptimal nonexact output')
    axes[0].bar(labels, strict, bottom=np.array(exact)+tied, color='#D55E00', label='Truth strictly excluded')
    for i, good in enumerate(exact):
        axes[0].text(i, good/2, str(good), ha='center', color='white', fontweight='bold')
        if tied[i]:
            axes[0].text(i, good+tied[i]/2, str(tied[i]), ha='center')
        if strict[i]:
            axes[0].text(i, good+tied[i]+strict[i]/2, str(strict[i]), ha='center', color='white')
    axes[0].set_ylim(0, 70)
    axes[0].set_ylabel('Graphs out of 64')
    axes[0].set_title('All 128 archived arm outputs retained')
    axes[0].legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(0, -.1))
    bars = axes[1].bar(['Strict exclusion', 'Cooptimal failure'], [data['strict_failures'], data['cooptimal_failures']], color=['#D55E00', '#E69F00'], width=.58)
    for bar, value in zip(bars, [data['strict_failures'], data['cooptimal_failures']]):
        axes[1].text(bar.get_x()+bar.get_width()/2, value-.9, f'{value}/14', ha='center', color='white' if value==9 else 'black', fontweight='bold')
    axes[1].axhline(10, color='#333333', linestyle='--', linewidth=1, label='Strict-exclusion target ≥10')
    axes[1].set_ylim(0, 13)
    axes[1].set_ylabel('Triple90 failures')
    axes[1].set_title('9 strict exclusions: target falsified')
    axes[1].legend(frameon=False, fontsize=9, loc='upper left')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('H1bx: nine failures cannot be repaired by optimal-solution tie handling alone', fontsize=12)
    fig.text(.5, .018, 'Exact integer objectives and 128 certificates rechecked; no new labels or LPs. Five ties do not specify a successful tie rule.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .13, 1, .91])
    folder = ROOT/'assets/figures'
    png, svg = [folder/f'repeated_objective_gap.{suffix}' for suffix in ['png', 'svg']]
    fig.savefig(png, dpi=180, metadata={'Software': 'Matplotlib'})
    fig.savefig(svg, metadata={'Date': None})
    plt.close(fig)
    provenance = dict(study='H1bx', input=str(source.relative_to(ROOT)), input_sha256=sha(source),
                      script=str(Path(__file__).resolve().relative_to(ROOT)), script_sha256=sha(__file__),
                      matplotlib=matplotlib.__version__, numpy=np.__version__,
                      output_sha256={str(path.relative_to(ROOT)): sha(path) for path in [png, svg]})
    (folder/'repeated_objective_gap-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(provenance))


if __name__ == '__main__':
    main()
