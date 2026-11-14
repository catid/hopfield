"""Render README figures exclusively from the public, audited result exports.

Run: uv run --with matplotlib --with numpy python scripts/plot_public_results.py
"""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/figures'
INPUTS = {}


def read(name):
    p = ROOT / name
    INPUTS[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(OUT / f'{stem}.{ext}', dpi=190, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'axes.labelcolor': '#273449',
                         'text.color': '#17243a', 'axes.edgecolor': '#b7c1cf',
                         'xtick.color': '#273449', 'ytick.color': '#273449',
                         'svg.hashsalt': 'hopfield-public-v1'})
    h = read('results/research/ripper/heldout.json')
    f = read('results/research/foureyes/audit-heldout.json')
    fig, ax = plt.subplots(figsize=(11, 5.8))
    colors = ['#8997a9', '#de9456', '#16887d']
    groups = [17, 21]
    series = [('Original CRAM', 'original_general', h['rows']),
              ('+ rule-checking stop', 'energy_AND_valid', f['records']),
              ('Digit symmetry + repulsion', 'canonical_repulsive_sparse', h['rows'])]
    for j, (label, arm, rows) in enumerate(series):
        vals = [[r['solved'] / r.get('n', r.get('n_puzzles', 10000)) * 100
                 for r in rows if r['k'] == k and r['arm'] == arm] for k in groups]
        assert all(len(v) == 6 for v in vals), (arm, vals)
        means = [np.mean(v) for v in vals]
        x = np.arange(2) + (j - 1) * .25
        ax.bar(x, means, width=.22, color=colors[j], label=label, zorder=3)
        for xx, vv, mean in zip(x, vals, means):
            ax.scatter(xx + np.linspace(-.06, .06, 6), vv, s=17, color='#17243a', alpha=.65, zorder=4)
            ax.text(xx, max(vv) + 2.7, f'{mean:.1f}%', ha='center', weight='bold', fontsize=12)
    ax.set(ylim=(0, 100), ylabel='Solved puzzles (%)', xticks=[0, 1],
           xticklabels=['17 training puzzles', '21 training puzzles'])
    ax.grid(axis='y', alpha=.18, zorder=0)
    fig.suptitle('Better Sudoku energies, without retraining', x=.08, ha='left', fontsize=22, weight='bold')
    ax.set_title(f"+{h['primary_effect_pp']:.1f} points vs original CRAM  ·  95% CI {h['primary_ci95_pp'][0]:.1f}–{h['primary_ci95_pp'][1]:.1f}", loc='left', fontsize=13, pad=17, weight='normal')
    ax.legend(loc='upper left', ncol=1, frameon=False, fontsize=10)
    fig.text(.08, .015, 'Reserved test slice: 10,000 puzzles · six trained seeds (dots) · four chains · 262,144 moves/chain\nRule checking adds Sudoku knowledge. Same move cap does not imply equal realized compute.', fontsize=10, color='#526075')
    fig.subplots_adjust(left=.08, right=.98, bottom=.18, top=.81)
    save(fig, 'sudoku-heldout')

    s = read('results/research/ripper/sparse.json')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), gridspec_kw={'width_ratios': [1.25, 1]})
    for k, color in zip([10, 17, 21], ['#16887d', '#447bb7', '#de9456']):
        rows = sorted([r for r in s['cells'] if r['k'] == k], key=lambda r: r['n'])
        axes[0].plot([r['n'] for r in rows], [r['inference_s_speedup'] for r in rows], 'o-', lw=2.5, color=color, label=f'k = {k}')
    axes[0].axhline(1, color='#8997a9', ls='--', lw=1)
    axes[0].set(xscale='log', ylim=(0, 4.5), xticks=[128, 512, 2048], xticklabels=['128', '512', '2,048'], xlabel='Puzzles per timed call', ylabel='Warm inference speedup (×)')
    axes[0].xaxis.set_minor_locator(NullLocator())
    axes[0].legend(frameon=False, ncol=3, fontsize=10)
    axes[0].grid(alpha=.15)
    axes[1].axis('off')
    axes[1].text(.02, .72, f"{s['primary_geomean_warm_inference_speedup']:.2f}×", fontsize=52, weight='bold', color='#16887d')
    axes[1].text(.04, .57, 'geometric mean warm inference speedup', fontsize=12)
    axes[1].text(.04, .39, f"95% timing interval: {s['primary_ci95'][0]:.2f}–{s['primary_ci95'][1]:.2f}×\nIncluding search setup: {s['complete_geomean_speedup']:.2f}×", fontsize=12, linespacing=1.8)
    axes[1].text(.04, .12, '63 paired calls: identical states and boards\nSame transformed energies, two backends', fontsize=11, linespacing=1.7)
    fig.suptitle('Sparse search preserves the tested trajectories', x=.08, ha='left', fontsize=20, weight='bold')
    fig.text(.08, .015, 'RTX PRO 6000 Blackwell Max-Q · nine fixed model/batch cells · six warmed timing pairs per cell\nCompilation excluded; setup and inference reported separately. This is not the paper’s H200 training-speed claim.', fontsize=10, color='#526075')
    fig.subplots_adjust(left=.08, right=.99, top=.83, bottom=.23, wspace=.26)
    save(fig, 'sparse-speed')
    (OUT / 'provenance.json').write_text(json.dumps({'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'inputs_sha256': INPUTS, 'matplotlib': matplotlib.__version__, 'numpy': np.__version__}, indent=2) + '\n')


if __name__ == '__main__':
    main()
