"""Compact, tracked Loki evidence. Requires completed independent CPU audits."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / 'artifacts/research/20261114/loki'
DEST = ROOT / 'results/research/loki'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    h3 = json.loads((ART / 'summary.json').read_text())
    h3b = json.loads((ART / 'h3b/summary.json').read_text())
    assert h3['status'] == h3b['status'] == 'validated'
    rows = []
    hashes = {}
    for study, summary in [('H3', h3), ('H3b', h3b)]:
        for name, digest in summary['input_sha256'].items():
            assert sha(ROOT / name) == digest, name
            hashes[name] = digest
        paths = ART.glob('n*/**/result.json') if study == 'H3' else (ART / 'h3b').glob('*/*/s*/*.json')
        for path in sorted(paths):
            result = json.loads(path.read_text())
            if result.get('status') != 'complete' or 'records_sha256' not in result:
                continue
            records = path.parent / ('records.npz' if study == 'H3' else path.stem + '.npz')
            assert sha(records) == result['records_sha256']
            arrays = np.load(records)
            rows.append(dict(study=study, n=result.get('n', 10), arm=result['arm'],
                stage=result.get('cohort', result.get('stage')),
                budget=80000 if result.get('stage') == 'secondary' and path.stem == 'final' else 20000,
                seed=result.get('inference_seed', result.get('seed')),
                unique_inputs=len(arrays['n20']), chains=20,
                n20_correct=int(arrays['n20'].sum()), n1_correct=int(arrays['n1'].sum()),
                valid_chains=int(arrays['valid'].sum()),
                coverage_numerator=int(arrays['found'].sum()),
                coverage_denominator=int(arrays['denom'].sum()),
                wrong_stops=result['wrong_stops'], capped_chains=result['budget_failures'],
                mean_moves=result['mean_moves'], search_seconds=result['search_seconds'],
                model_sha256=result['model_sha256'], protocol_sha256=result['protocol_sha256'],
                records_sha256=result['records_sha256'], result_path=str(path.relative_to(ROOT)),
                result_sha256=sha(path)))
    order = {'adamw-decay': 0, 'adamw-lr0.3': 0, 'adam-nodecay': 1, 'sgd-pre': 2, 'sgd-plain': 3}
    rows.sort(key=lambda r: (r['study'], r['n'], r['stage'], order.get(r['arm'], 1), r['arm'], r['budget'], r['seed']))
    with (DEST / 'per_seed.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = dict(status='validated', export_source_sha256=sha(__file__),
        H3={key: h3[key] for key in ['decision', 'selected_arm', 'rows', 'comparisons', 'distinct_models', 'fit_updates_each', 'search_jobs', 'independent_board_validation', 'inference_seed_note']},
        H3b={key: h3b[key] for key in ['decision', 'rows', 'primary_comparisons', 'secondary_budget_tradeoffs', 'limits']},
        fit_summaries={arm: {key: value for key, value in fit.items() if key != 'offsets'} for arm, fit in h3b['fit_summaries'].items()},
        provenance=dict(audits={str(path.relative_to(ROOT)): sha(path) for path in [ART / 'summary.json', ART / 'h3b/summary.json']},
                        source_model_protocol_and_outcome_sha256=hashes,
                        per_seed_sha256=sha(DEST / 'per_seed.csv')))
    (DEST / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    (DEST / 'README.md').write_text('''# Loki: optimizer and weight-decay experiments

The fixed-learning-rate weight-decay hypothesis was supported on 6,513 unique official n=10 test inputs with 20 chains and three paired search seeds. BP AdamW (decay 0.01) reached 99.605916% N20 accuracy; setting decay to zero reached 100%, and preconditioned SGD reached 99.984646%. Coverage was 92.283227%, 93.559127%, and 93.585165%, respectively.

The original AdamW-versus-preconditioned-SGD hypothesis was falsified on its disjoint development confirmation cohort. H3b then tested the decay intervention at the already-selected learning rate 0.3. It used the same 97,554 fitting rows, 4,000 full-batch updates and normalized inference temperature. All models differentiate the local conditional loss; BP does not unroll search.

The no-decay accuracy difference versus preconditioned SGD was +0.015354 percentage points, with one-sided 95% endpoints [0, 0.046062]. Coverage difference was -0.026039 points, endpoints [-0.206310, 0.142826]. The conservative exact regression bound was 0.045986%, below the preregistered 0.1-point margin. Search seeds are repeated draws within inputs, not independent training seeds.

[summary.json](summary.json) includes decisions, aggregate results, uncertainty, source/model/protocol/outcome hashes and limits. [per_seed.csv](per_seed.csv) includes integer success and coverage counts for every search, plus hashes. The larger saved boards and checkpoints remain in the local artifact archive. These files do not assert an optimizer advantage across tasks or a hardware speedup.

The separate 1,024-input budget control captures 20,000 and 80,000 moves from the same trajectory. The AdamW-decay accuracy changed from 99.348958% to 99.479167%; no-decay Adam and preconditioned SGD remained at 100%. This fixed canonical subset is not a random representative sample of the full test. Snapshot parity was verified against the upstream solver; multi-block snapshot capture is rejected because shared RNG state couples blocks.

Protocols and full reports: [H3](../../../research/20261114/loki/REPORT.md), [H3b](../../../research/20261114/loki/h3b/REPORT.md). Regenerate after independent audits with `upstream/.venv/bin/python research/20261114/loki/export_public.py`.
''')
    print(json.dumps(dict(status='exported', rows=len(rows), hashes=len(hashes), destination=str(DEST))))


if __name__ == '__main__':
    main()
