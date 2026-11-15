"""Deterministic H1bt conditional-randomization p-value display."""
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
    source = ROOT/'results/research/ripper/signal_detection/summary.json'
    data = json.loads(source.read_text())
    plt.rcParams.update({'font.size': 10, 'svg.hashsalt': 'h1bt-signal-detection-v1'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    colors = {'uniform': '#0072B2', 'balanced': '#D55E00'}
    unresolved = []
    for ax, cohort, title in zip(axes, ['real', 'null'], ['Examined real banks (retrospective)', 'Fresh IID null banks (prospective)'], strict=True):
        for ai, arm in enumerate(['uniform', 'balanced']):
            rows = [r for r in data['rows'] if r['cohort'] == cohort and r['arm'] == arm]
            valid = [r for r in rows if r['test']['numerator'] is not None]
            unresolved += [dict(cohort=cohort, arm=arm, bank=r['bank']) for r in rows if r['test']['numerator'] is None]
            ax.scatter([r['bank']+1+(ai-.5)*.22 for r in valid], [r['test']['numerator']/32 for r in valid],
                       s=27, marker=['o', '^'][ai], color=colors[arm], alpha=.85, label=arm.capitalize(), zorder=3)
        endpoint = data['endpoints'][cohort]
        threshold = 'target ≥29' if cohort == 'real' else 'target ≤2'
        count = f"{endpoint['lower']}/32" if endpoint['lower'] == endpoint['upper'] else f"{endpoint['lower']}–{endpoint['upper']}/32"
        ax.set_title(f'{title}\nJoint signals: {count}; {threshold} — {endpoint["decision"]}', fontsize=10)
        ax.set_yscale('log', base=2)
        ax.axhline(1/32, color='#333333', linestyle='--', linewidth=1, label='Per-arm threshold 1/32')
        ax.set_ylim(.025, 1.16)
        ax.set_xlim(.2, 32.8)
        ax.set_xticks([1, 8, 16, 24, 32])
        ax.set_xlabel('Bank')
        ax.set_yticks([1/32, 1/16, 1/8, 1/4, 1/2, 1], ['1/32', '1/16', '1/8', '1/4', '1/2', '1'])
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Conditional randomization p-value')
    axes[0].legend(loc='upper left', frameon=False, fontsize=9)
    fig.suptitle('H1bt: detecting association with 31 marginal-preserving shuffles', fontsize=13)
    fig.text(.5, .025, 'Both arms must pass. Conservative ties; exact integer comparisons. Signal detection does not certify graph recovery.', ha='center', fontsize=9)
    if unresolved:
        fig.text(.5, .065, f'{len(unresolved)} unresolved arm tests are retained in endpoint bounds and omitted from p-value coordinates.', ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .09, 1, .92])
    destination = ROOT/'assets/figures'
    destination.mkdir(parents=True, exist_ok=True)
    png = destination/'signal_detection.png'
    svg = destination/'signal_detection.svg'
    fig.savefig(png, dpi=180, metadata={'Software': 'Matplotlib'})
    fig.savefig(svg, metadata={'Date': None})
    plt.close(fig)
    provenance = dict(study='H1bt', input=str(source.relative_to(ROOT)), input_sha256=sha(source),
                      script=str(Path(__file__).resolve().relative_to(ROOT)), script_sha256=sha(__file__),
                      matplotlib=matplotlib.__version__, numpy=np.__version__, unresolved_arms=unresolved,
                      output_sha256={str(path.relative_to(ROOT)): sha(path) for path in [png, svg]})
    (destination/'signal_detection-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(provenance))


if __name__ == '__main__':
    main()
