"""Render H3b/H3c solely from compact tracked public exports."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEST = ROOT / 'results/research/loki'
COLORS = {'adamw-decay': '#c44e52', 'adam-nodecay': '#2878b5', 'sgd-pre': '#ca8a04'}


def main():
    sources = [DEST / 'summary.json', DEST / 'per_seed.csv', DEST / 'h3c/summary.json', DEST / 'h3c/per_seed.csv', DEST / 'h3d/summary.json']
    base = json.loads(sources[0].read_text()); curve = json.loads(sources[2].read_text())
    selection = json.loads(sources[4].read_text())
    assert base['status'] == curve['status'] == selection['status'] == 'validated'
    with sources[1].open() as handle: seeds = list(csv.DictReader(handle))
    with sources[3].open() as handle: prefix = list(csv.DictReader(handle))
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.6), gridspec_kw={'width_ratios': [1.1, 1.2, 1]})
    fig.subplots_adjust(left=.055, right=.985, bottom=.27, top=.77, wspace=.30)
    fig.suptitle('N-Queens: removing weight decay and shortening the fit', x=.055, y=.965,
                 ha='left', fontsize=19, fontweight='bold')
    fig.text(.055, .89, 'n=10 · 6,513 unique inputs · 20 chains · 3 search seeds · cap 20,000 · original lowest-energy output selection', fontsize=11)
    arms = ['adamw-decay', 'adam-nodecay', 'sgd-pre']
    labels = ['BP AdamW\ndecay 0.01', 'BP Adam\nno decay', 'SGD +\npreconditioner']
    ax = axes[0]
    for i, arm in enumerate(arms):
        data = [r for r in seeds if r['study'] == 'H3b' and r['stage'] == 'primary' and r['arm'] == arm]
        assert len(data) == 3
        errors = np.array([int(r['unique_inputs']) - int(r['n20_correct']) for r in data])
        ax.bar(i, errors.mean(), color=COLORS[arm], width=.64, alpha=.78)
        ax.scatter(i + np.linspace(-.12, .12, 3), errors, color='black', s=20, zorder=4)
        ax.annotate(f'{errors.mean():.2f}', (i, max(errors)), xytext=(0, 8), textcoords='offset points', ha='center')
    ax.set_xticks(range(3), labels); ax.set_ylim(0, 31)
    ax.set_ylabel('Failed inputs out of 6,513'); ax.set_title('A   Weight decay at 4,000 updates', loc='left', fontweight='bold', pad=16)
    ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
    ax = axes[1]
    for arm in ['adam-nodecay', 'sgd-pre']:
        xs = [250, 500, 1000, 2000, 4000]; ys = []
        for updates in xs:
            data = [r for r in prefix if r['arm'] == arm and int(r['updates']) == updates]
            assert len(data) == 3
            errors = np.array([int(r['unique_inputs']) - int(r['n20_correct']) for r in data])
            ys.append(errors.mean())
            ax.scatter(np.repeat(updates, 3), errors, color=COLORS[arm], alpha=.5, s=14, zorder=4)
        ax.plot(xs, ys, color=COLORS[arm], marker='o', label='BP Adam, no decay' if arm == 'adam-nodecay' else 'SGD + preconditioner')
    ax.axvline(1000, color='#888', linestyle=':', linewidth=1, zorder=0)
    ax.text(1000, 28.8, 'Frozen primary\nAdam checkpoint', ha='center', va='top', fontsize=9)
    ax.set_xscale('log', base=2); ax.set_xticks(xs, [f'{x:,}' for x in xs]); ax.set_ylim(0, 31)
    ax.set_xlabel('Prefix of the unchanged 4,000-update schedule')
    ax.set_ylabel('Failed inputs out of 6,513'); ax.grid(axis='y', alpha=.18)
    ax.set_title('B   Every fixed training prefix', loc='left', fontweight='bold', pad=16)
    ax.legend(loc='upper right', bbox_to_anchor=(1.04, .62), fontsize=8.5, frameon=False)
    ax = axes[2]
    for i, arm in enumerate(['adam-nodecay', 'sgd-pre']):
        data = curve['timing']['samples_seconds'][arm]
        ax.bar(i, np.median(data), color=COLORS[arm], width=.58, alpha=.78)
        ax.scatter(i + np.linspace(-.10, .10, len(data)), data, color='black', s=20, zorder=4)
        ax.annotate(f'{np.median(data):.2f} s', (i, max(data)), xytext=(0, 8), textcoords='offset points', ha='center')
    ax.set_xticks([0, 1], ['BP Adam, no decay\n1,000 updates', 'SGD + preconditioner\n4,000 updates'])
    ax.set_ylabel('Complete in-memory fit (seconds)'); ax.set_ylim(0, 9.6)
    ax.set_title('C   Complete fitting time', loc='left', fontweight='bold', pad=16)
    ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
    ratio = curve['primary']['median_complete_fit_time_ratio']
    fig.text(.055, .12, f"Frozen H3c result: {curve['decision']}. Adam at 1,000 vs SGD at 4,000: same observed N20 accuracy, {abs(curve['primary']['coverage_difference_pp']):.3f} pp lower coverage, fitting-time ratio {ratio:.3f}.", fontsize=10.5)
    fig.text(.055, .065, 'Dots show every repeated search seed (A–B) or timed fit (C), not confidence intervals. One deterministic fitted trajectory per optimizer.\nAdaptive follow-up on a previously used official test set. Fit timings use one Loki Blackwell GPU, three repeats after warmup; inference time is separate.', fontsize=9, color='#444', linespacing=1.5)
    fig.text(.055, .018, f"Later H3d reanalysis: prioritizing learned stopped flags recovers decay-AdamW's {selection['primary']['recovered_inputs']} input–seed failures from the same searches (a derivative output policy).", fontsize=9, color='#444')
    for ext in ['png', 'svg']:
        fig.savefig(DEST / f'weight-decay-and-prefix.{ext}', dpi=180)
    receipt = dict(plot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        outputs={f'weight-decay-and-prefix.{ext}': hashlib.sha256((DEST / f'weight-decay-and-prefix.{ext}').read_bytes()).hexdigest() for ext in ['png', 'svg']})
    (DEST / 'plot-provenance.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt['outputs']))


if __name__ == '__main__': main()
