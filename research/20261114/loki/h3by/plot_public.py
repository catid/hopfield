"""Deterministic fresh-cohort total-call figure with the prespecified target."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[4]
P = ROOT/'results/research/loki/h3by'


def main():
    source = P/'summary.json'
    s = json.loads(source.read_text())
    assert s['status'] == 'validated'
    B, D = s['baseline_policy_calls'], s['deeper_policy_calls']
    saving = 100*(B-D)/B
    plt.rcParams.update({'font.size': 12, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'h3by-uniform-truth-prior-shift'})
    fig, ax = plt.subplots(figsize=(12.8, 9))
    fig.subplots_adjust(left=.12, right=.94, bottom=.34, top=.72)
    fig.suptitle(f"Uniform-prior shift: {abs(saving):.2f}% {'fewer' if saving >= 0 else 'more'} membership calls",
                 x=.04, y=.96, ha='left', fontsize=20, fontweight='bold')
    fig.text(.04, .895, f"PRIMARY {s['decision'].upper()}: {D:,} versus {B:,}; frozen target ≤{s['primary']['passing_maximum']:,}.",
             fontsize=13, fontweight='bold')
    fig.text(.04, .82, 'All 4,096 fresh uniform-truth pairs completed; both arms execute and pay separately.\n'
             f"All exact flags agree: {s['exact_cases']:,} identified cases and {s['retained_failures']} retained failures.",
             fontsize=11.5, linespacing=1.5)
    arms = ['baseline', 'deeper']
    maps = [s['MAP_calls'][arm] for arm in arms]
    trees = [s['tree_calls'][arm] for arm in arms]
    ax.bar([0, 1], maps, width=.5, color='#557f9c', label='MAP calls')
    ax.bar([0, 1], trees, bottom=maps, width=.5, color='#bb805f', label='Tree calls')
    ax.set_xticks([0, 1], ['Original four-MAP / depth-four policy\nnew, separately paid baseline',
                          'Fixed deeper handoff policy\ndepth = 8 − used MAP calls'])
    ax.set_ylabel('Total actual membership calls')
    ax.set_ylim(0, max(B, D)*1.25)
    ax.axhline(s['primary']['passing_maximum'], color='#744848', ls='--', lw=1.2)
    ax.text(.5, s['primary']['passing_maximum']*.935,
            f"Passing target ≤{s['primary']['passing_maximum']:,}", ha='center', fontsize=10.5,
            color='#744848', bbox=dict(facecolor='white', edgecolor='none', pad=2))
    ax.legend(loc='upper left', frameon=False, ncol=2, bbox_to_anchor=(0, 1.04))
    ax.grid(axis='y', alpha=.13)
    ax.set_axisbelow(True)
    for i, total in enumerate([B, D]):
        ax.text(i, total+300, f'{total:,}', ha='center', fontsize=13, fontweight='bold')
    fig.text(.04, .25, f"{B-D:,} calls saved; all {B+D:,} actual calls across both arms are charged.",
             fontsize=12.3, fontweight='bold')
    fig.text(.04, .18, f"{s['independently_paid_cross_arm_reacquisitions']:,} shared-board queries independently reacquired; "
             f"{s['source_cases_with_prior_same_study_duplicates']} within-cohort duplicate cases retained.\n"
             'One shared eight-positive source bank per pair: 32,768 new positive appearances.',
             fontsize=11.2, linespacing=1.5)
    fig.text(.04, .095, 'Uniform truth sampling; both policies keep fitted hierarchical weights. No actual-prior expectation theorem.\n'
             'Full proof archive retained outside Git; compact checkout verification has explicit proof-replay limits.',
             fontsize=10.7, color='#444', linespacing=1.5)
    fig.text(.04, .03, f"BP AdamW: not measured. {s['aggregate_CPU_seconds']:.3f} CPU s / "
             f"{s['wall_seconds']:.3f} wall s; 0 GPU s. Post-run auditing/export timed separately.",
             fontsize=10.5, color='#444')
    for ext in ['png', 'svg']:
        fig.savefig(P/f'fresh-paired-total-cost.{ext}', dpi=180, facecolor='white',
                    metadata={'Date': None} if ext == 'svg' else None)
    receipt = dict(input_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in [source, Path(__file__)]},
                   output_sha256={f'fresh-paired-total-cost.{ext}': hashlib.sha256(
                                  (P/f'fresh-paired-total-cost.{ext}').read_bytes()).hexdigest() for ext in ['png', 'svg']})
    (P/'plot-provenance.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(receipt['output_sha256'])


if __name__ == '__main__':
    main()
