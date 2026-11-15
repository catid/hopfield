"""Stage H2bm independently, then publish only after shared-file release."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
DEST = ROOT / 'artifacts/research/20261114/work/bp_geometry_decomposition'
STAGED = DEST / 'public-staging'
PUBLIC = ROOT / 'results/research/work/bp-geometry-decomposition'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def evidence():
    STAGED.mkdir(exist_ok=True)
    for name in ['protocol.json', 'synthetic-gate.json']:
        shutil.copyfile(DEST / name, STAGED / name)
    for name in ['analysis.json', 'per-model-input.csv', 'fixture-checks.csv']:
        shutil.copyfile(DEST / 'analysis' / name, STAGED / name)
    save(STAGED / 'source-reference.json', {
        'source_directory_from_public_package': '../cross-certificate-screen',
        'full_failure_fixture_count': 510, 'fixed_control_fixture_count': 48,
        'source_files_sha256': json.loads((DEST / 'protocol.json').read_text())['sealed_source_sha256'],
        'scope': 'Reuses complete published H2bl source fixtures by hash; no duplicated full arrays.'})


def publish():
    result = json.loads((STAGED / 'analysis.json').read_text())
    check = json.loads((STAGED / 'public-verification.json').read_text())
    fig = json.loads((STAGED / 'figure-determinism.json').read_text())
    assert result['status'] == check['status'] == fig['status'] == 'passed'
    assert fig['visually_inspected'] and not PUBLIC.exists()
    root = PUBLIC.parent
    summary = json.loads((root / 'summary.json').read_text())
    assert 'bp_geometry_decomposition' not in summary
    shutil.copytree(STAGED, PUBLIC)
    save(root / 'bp-geometry-decomposition-summary.json', result)
    summary['bp_geometry_decomposition'] = {
        'hypothesis': 'H2bm', 'fresh_inputs': 0, 'new_scientific_evaluations': 0,
        'source_model_inputs': 12288, 'partition': result['partition'],
        'sole_endpoint': result['sole_endpoint'],
        'retention': 'Definitional shortest-implies-simple-path invariant, not a scientific endpoint.',
        'scope': 'Analysis of sealed H2bl fields; no fresh accuracy, inference speed or executed cascade claim.'}
    summary['hypotheses'].append({'campaign': 'bp_geometry_decomposition',
        'hypothesis': 'H2bm_invalid_geometry_detects90', 'outcome': result['sole_endpoint']})
    p = json.loads((DEST / 'protocol.json').read_text())
    summary['protocols']['bp-geometry-decomposition-protocol'] = {
        'sha256': sha(DEST / 'protocol.json'), 'frozen_utc': p['frozen_utc'],
        'hypothesis': 'H2bm', 'status': 'completed'}
    summary['audit_sha256'][str((DEST / 'analysis/analysis.json').relative_to(ROOT))] = sha(DEST / 'analysis/analysis.json')
    summary['updated_utc'] = datetime.now(timezone.utc).isoformat()
    save(root / 'summary.json', summary)
    save(root / 'manifest.json', {'source_sha256': sha(Path(__file__)),
         'files': {str(q.relative_to(root)): sha(q) for q in sorted(root.rglob('*'))
                   if q.is_file() and q != root / 'manifest.json'}})


if __name__ == '__main__':
    publish() if '--publish' in sys.argv else evidence()
