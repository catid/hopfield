"""Plot tracked H3x structural evidence and explicitly inherited search outcomes."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[4]
DEST = ROOT / 'results/research/loki/h3x'
def main():
    source = DEST/'summary.json'
    summary = json.loads(source.read_text())
    assert summary['status'] == 'validated'
    result = summary['source_result']
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'svg.fonttype':'none', 'svg.hashsalt':'cram-h3x-public'})
    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    fig.subplots_adjust(left=.065, right=.98, bottom=.43, top=.75, wspace=.26)
    fig.suptitle('A complete source universe identifies one minimal finite rule', x=.065, y=.96,
                 ha='left', fontsize=18, fontweight='bold')
    fig.text(.065, .89, 'Conditional hypothesis supported: unique generator (1, 3), with exact N13 attack support.',
             fontsize=12, fontweight='bold')
    fig.text(.065, .83, '176 complete source positives + supplied completeness → 14,024 negatives · all 32 candidate subsets certified', fontsize=10.7)
    labels = ['(1, 3)', '(2, 8)', '(2, 11)', '(8, 9)', '(10, 11)', 'All four\nother rules']
    other = result['subsets'][30]
    covered = [o['source_negative_coverage'] for o in result['orbits']] + [other['rejected_negatives']]
    for k, value in enumerate(covered):
        height = value/result['source_negatives']*100
        axes[0].bar(k, height, color='#19876b' if k == 0 else '#8799ad', width=.65)
        axes[0].annotate(f'{value:,}', (k, height), xytext=(0, 6), textcoords='offset points', ha='center', fontsize=8.8)
    axes[0].axhline(100, color='#444', ls=':', lw=1.2)
    axes[0].set_xticks(range(6), labels, fontsize=9)
    axes[0].set_ylabel('Source negatives rejected (%)')
    axes[0].set_title('A   Single D4 orbit generators and their union', loc='left', pad=12, fontweight='bold')
    colors = ['#c44e52', '#19876b', '#8c5eb5']
    for k, row in enumerate(summary['rows']):
        for shift, key, alpha, hatch in [(-.17, 'certified_success_pct', .9, None), (.17, 'certificate_first_accuracy_pct', .35, '//')]:
            height = row[key]
            axes[1].bar(k+shift, height, color=colors[k], width=.3, alpha=alpha, hatch=hatch)
            axes[1].annotate(f'{height:.2f}', (k+shift, height), xytext=(0, -14), textcoords='offset points', ha='center', fontsize=8.4,
                             bbox=dict(facecolor='white', edgecolor='none', pad=.1))
    axes[1].set_xticks(range(3), ['BP AdamW\nH3w reference', 'Uniform filtered\nH3w reference', 'Minimum rule\nmapped H3w oracle'], fontsize=9)
    axes[1].set_ylabel('Valid completion accuracy (%)')
    axes[1].set_title('B   Previously measured H3w searches', loc='left', pad=12, fontweight='bold')
    for ax in axes:
        ax.set_ylim(0, 114)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor='#777', alpha=.9, label='Certified primary'), Patch(facecolor='#777', alpha=.35, hatch='//', label='Certificate-first with energy fallback')],
               loc='lower left', bbox_to_anchor=(.56, .30), ncol=1, frameon=False, fontsize=9)
    fig.text(.065, .325, '16 rule subsets fit the full source; sparsity selects one.\nThe other four rules together still miss 1,596 negatives.', fontsize=10.3, linespacing=1.5)
    fig.text(.065, .235, 'The inferred lines plus the selected finite orbit produce all 3,808 true N13 attack pairs, with zero false or missed pairs.\nThe source model arrays and H3w oracle arrays are byte-identical; backend, calibration, inputs and search seeds also match.', fontsize=10, linespacing=1.5)
    fig.text(.065, .15, 'Added priors: closed-world source completeness, D4 symmetry and minimum-cardinality rules. Completeness alone leaves\n16 indistinguishable rule sets. This is not identification from positives alone or a reproduction of the paper.', fontsize=9.7, color='#444', linespacing=1.5)
    fig.text(.065, .065, 'Source selection + independent certificate: 3.545 CPU seconds. Zero new GPU searches. Mapped results reuse 4,096 H3w inputs\nand three search seeds; 245,760 inherited saved states re-audited. This mapping is not an independent empirical replication.', fontsize=9.4, color='#444', linespacing=1.5)
    for ext in ['png', 'svg']:
        fig.savefig(DEST/f'minimum-rule.{ext}', dpi=180, facecolor='white', metadata={'Date':None} if ext == 'svg' else None)
    receipt = dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source, Path(__file__)]},
                   output_sha256={f'minimum-rule.{ext}':hashlib.sha256((DEST/f'minimum-rule.{ext}').read_bytes()).hexdigest() for ext in ['png', 'svg']})
    (DEST/'plot-provenance.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt['output_sha256']))

if __name__ == '__main__':
    main()
