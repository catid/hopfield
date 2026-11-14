"""Export audited H4 measurements, without rerunning inference."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / 'artifacts/research/20261114/foureyes'
OUT = ROOT / 'results/research/foureyes'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {'scope': 'Post-training Sudoku stopping interventions; additional task-rule knowledge in validity gate. Same move cap, not equal realized compute.', 'studies': []}
    rows = []
    for phase, filename in [('discovery', 'audit.json'), ('confirmation', 'confirmation-audit.json'), ('calibration', 'calibration/audit.json'), ('heldout', 'heldout-audit.json')]:
        path = RAW / filename
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        assert d['status'] == 'validated'
        (OUT / f'audit-{phase}.json').write_text(json.dumps(d, indent=2, allow_nan=False) + '\n')
        summary['studies'].append({'phase': phase, 'comparisons': d['comparisons'], 'saved_boards_verified': d.get('saved_boards_checked', d.get('saved_boards')), 'source': str(path.relative_to(ROOT)), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        rows.extend(dict(phase=phase, **r) for r in d.get('records', d.get('rows', [])))
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / 'per_seed.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Exported {len(rows)} measured rows from {len(summary["studies"])} audits')


if __name__ == '__main__':
    main()
