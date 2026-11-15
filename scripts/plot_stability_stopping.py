"""Deterministic H1bu accuracy and virtual stopping-budget comparison."""
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
    source = ROOT/'results/research/ripper/stability_stopping/summary.json'
    data = json.loads(source.read_text())
    plt.rcParams.update({'font.size': 10, 'svg.hashsalt': 'h1bu-stability-stopping-v1'})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    x = [group['rows'] for group in data['grouped']]
    y = [group['success'] for group in data['grouped']]
    axes[0].plot(x, y, 'o-', color='#0072B2', linewidth=2)
    for n, good in zip(x, y):
        axes[0].annotate(f'{good}/32', (n, good), xytext=(0, 9), textcoords='offset points', ha='center')
    accuracy = data['endpoints']['stopped_exact']
    axes[0].set_title(f'Exact graph recovery\nStopped: {accuracy["lower"]}/32; target ≥31 — {accuracy["decision"]}', fontsize=10)
    axes[0].set_xlabel('Literal prefix rows')
    axes[0].set_ylabel('Exact graphs out of 32')
    axes[0].set_xticks([90, 180, 270, 360])
    axes[0].set_yticks([0, 8, 16, 24, 32])
    axes[0].set_xlim(70, 380)
    axes[0].set_ylim(-2, 37)
    budget = data['endpoints']['virtual_rows']
    mean = budget['total_rows']/32
    bars = axes[1].bar(['Fixed 270', 'Stability stop', 'Fixed 360'], [270, mean, 360], color=['#999999', '#D55E00', '#999999'], width=.62)
    for bar, value in zip(bars, [270, mean, 360]):
        axes[1].text(bar.get_x()+bar.get_width()/2, value+8, f'{value:g}', ha='center')
    axes[1].axhline(270, color='#333333', linestyle='--', linewidth=1, label='Stopping target ≤270')
    axes[1].set_title(f'Virtual deployment cost\nMean stopping rows: {mean:g} — {budget["decision"]}', fontsize=10)
    axes[1].set_ylabel('Mean rows per bank')
    axes[1].set_ylim(0, 410)
    axes[1].legend(frameon=False, fontsize=9, loc='upper left')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    hist = data['stop_histogram']
    fig.suptitle('H1bu: exact stopped graphs, but the row-budget target was missed', fontsize=12)
    fig.text(.5, .055, f'Stops: {hist["180"]} at 180, {hist["270"]} at 270, {hist["360"]} at 360. All 32 banks retained; no early wrong stable graph.', ha='center', fontsize=9)
    fig.text(.5, .018, 'The experiment physically measured all 360 rows per bank (829,440 queries). Virtual stopping used 648,000 queries.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .105, 1, .92])
    folder = ROOT/'assets/figures'
    folder.mkdir(parents=True, exist_ok=True)
    png, svg = [folder/f'stability_stopping.{suffix}' for suffix in ['png', 'svg']]
    fig.savefig(png, dpi=180, metadata={'Software': 'Matplotlib'})
    fig.savefig(svg, metadata={'Date': None})
    plt.close(fig)
    provenance = dict(study='H1bu', input=str(source.relative_to(ROOT)), input_sha256=sha(source), script=str(Path(__file__).resolve().relative_to(ROOT)),
                      script_sha256=sha(__file__), matplotlib=matplotlib.__version__, numpy=np.__version__,
                      output_sha256={str(path.relative_to(ROOT)): sha(path) for path in [png, svg]})
    (folder/'stability_stopping-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(provenance))


if __name__ == '__main__':
    main()
