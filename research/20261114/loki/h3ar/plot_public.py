"""Show the source witness explanation beside exact target capacity."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[4]; PUBLIC = ROOT/'results/research/loki/h3ar'


def main():
    paths = [PUBLIC/'summary.json', PUBLIC/'source-selection.json'] + [PUBLIC/f'b{i}/source-evidence.npz' for i in range(8)]
    summary = json.loads(paths[0].read_text()); source = json.loads(paths[1].read_text())
    assert summary['status'] == 'validated'
    counts = np.stack([np.load(p)['sample_orbit_incidence'].sum(0) for p in paths[2:]])
    plt.rcParams.update({'font.size': 10.5, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none', 'svg.hashsalt': 'cram-h3ar-public'})
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.7), gridspec_kw={'width_ratios': [1.3, 1]})
    fig.subplots_adjust(left=.075, right=.97, bottom=.31, top=.76, wspace=.3)
    fig.suptitle('A supplied rule vocabulary resolves these eight 16-label banks', x=.065, y=.95, ha='left', fontsize=17, fontweight='bold')
    fig.text(.065, .88, 'SUPPORTED: all eight ML selections are unique and exact; the consistency control gives the identical graph', fontsize=11.3, fontweight='bold')
    fig.text(.065, .82, '56 supplied models: ordinary queens plus 0 / 1 / 2 D4 leaper orbits with 1 ≤ a < b ≤ 5; no BP fits', fontsize=10.8)
    axes[0].imshow(counts, cmap='YlGnBu', vmin=0, vmax=16, aspect='auto')
    for i in range(8):
        for j in range(10): axes[0].text(j, i, str(counts[i, j]), ha='center', va='center', color='white' if counts[i, j] >= 9 else '#222', fontsize=9.1)
    axes[0].set_xticks(range(10), ['1,2', '1,3', '1,4', '1,5', '2,3', '2,4', '2,5', '3,4', '3,5', '4,5'], rotation=45)
    axes[0].set_yticks(range(8), ['Z0', 'Z1', 'Z2', 'Z3', 'L0', 'L1', 'L2', 'L3'])
    axes[0].set_xlabel('Supplied D4 orbit (a,b)'); axes[0].set_title('A   Positives witnessing each orbit (out of 16)', loc='left', fontweight='bold', pad=14)
    for bank in source['banks']:
        index = bank['index']; a = next(r for r in summary['rows'] if r['index'] == index and r['primary']); b = next(r for r in summary['rows'] if r['index'] == index and not r['primary'])
        axes[1].bar(index-.17, a['capacity_pct'], .32, color='#226f68', label='ML selection' if index == 0 else None)
        axes[1].bar(index+.17, b['capacity_pct'], .32, color='#91b7b2', label='Consistency envelope' if index == 0 else None)
    axes[1].axhline(95, color='#ad4841', ls='--', lw=1.2); axes[1].set_ylim(0, 112)
    axes[1].set_xticks(range(8), ['Z0', 'Z1', 'Z2', 'Z3', 'L0', 'L1', 'L2', 'L3'])
    axes[1].set_ylabel('Exact valid-input capacity (%)'); axes[1].set_title('B   Both constructions reach 100%; zero invalid states', loc='left', fontweight='bold', pad=14, fontsize=10.8)
    axes[1].grid(axis='y', alpha=.17); axes[1].set_axisbelow(True); axes[1].legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.13), ncol=2, fontsize=9)
    fig.text(.065, .2, 'Every bank witnesses all nine false orbit prohibitions. Only its true orbit remains unobserved.', fontsize=11, fontweight='bold')
    fig.text(.065, .135, 'The uniform-without-replacement likelihood selects the true model, but the consistency envelope already does so.\nThese fixed-bank successes do not establish a likelihood-normalization advantage or 16-label population reliability.', fontsize=10.7, linespacing=1.45)
    fig.text(.065, .055, f"All 56 complete N12 candidate sets and both full N13 supports independently checked; no target exclusions.\nStudy: {summary['aggregate_CPU_seconds']:.3f} CPU s / {summary['wall_seconds']:.3f} wall s; 0 GPU s. Strong queen, D4, radius and model-size priors remain explicit.", fontsize=10.2, color='#444', linespacing=1.4)
    for ext in ('png', 'svg'): fig.savefig(PUBLIC/f'finite-likelihood.{ext}', dpi=180, facecolor='white', metadata={'Date': None} if ext == 'svg' else None)
    receipt = dict(input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[Path(__file__)]}, output_sha256={f'finite-likelihood.{ext}': hashlib.sha256((PUBLIC/f'finite-likelihood.{ext}').read_bytes()).hexdigest() for ext in ('png', 'svg')})
    (PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt, indent=2)+'\n'); print(receipt['output_sha256'])


if __name__ == '__main__': main()
