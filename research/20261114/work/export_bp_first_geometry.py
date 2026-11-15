"""H2bn post-audit publication with zero scientific-count increment."""
import json
from pathlib import Path
import shutil
import sys
import study as S

DEST = S.OUT / 'bp_first_geometry_cascade'
PUBLIC = S.ROOT / 'results/research/work/bp-first-geometry-cascade'


def evidence():
    assert not PUBLIC.exists()
    shutil.copytree(DEST / 'public-staging', PUBLIC)


def publish():
    a = json.loads((PUBLIC / 'audit.json').read_text())
    v = json.loads((PUBLIC / 'public-verification.json').read_text())
    f = json.loads((PUBLIC / 'figure-determinism.json').read_text())
    assert a['status'] == v['status'] == f['status'] == 'passed' and f['visually_inspected']
    root = PUBLIC.parent
    summary = json.loads((root / 'summary.json').read_text())
    assert 'bp_first_geometry_cascade' not in summary
    summary['bp_first_geometry_cascade'] = {
        'hypothesis': 'H2bn', 'fresh_inputs': 0, 'new_scientific_evaluations': 0,
        'source_model_inputs': 12288, 'paired_full_cohort_replays': 4,
        'source_preservation': a['source_preservation'], 'cost_endpoint': a['cost_endpoint'],
        'scope': 'Actual BP-first geometry-routed LP96 source replays. Complete measured answer-production cost includes independent readiness/input/hidden/solver/geometry/certificate/return/mapping, excludes scientific diagnostics/archive/audit and barrier waits. No fresh accuracy/population claim.'}
    for name, value in [('source_preservation', a['source_preservation']), ('complete_answer_cost_saving10', a['cost_endpoint'])]:
        summary['hypotheses'].append({'campaign': 'bp_first_geometry_cascade', 'hypothesis': 'H2bn_' + name, 'outcome': value})
    p = json.loads((PUBLIC / 'protocol.json').read_text())
    summary['protocols']['bp-first-geometry-cascade-protocol'] = {'sha256': S.sha(PUBLIC / 'protocol.json'),
        'frozen_utc': p['frozen_utc'], 'hypothesis': 'H2bn', 'status': 'completed'}
    summary['audit_sha256'][str((DEST / 'audit.json').relative_to(S.ROOT))] = S.sha(DEST / 'audit.json')
    summary['updated_utc'] = S.utc()
    S.save(root / 'bp-first-geometry-cascade-summary.json', a)
    S.save(root / 'summary.json', summary)
    S.save(root / 'manifest.json', {'source_sha256': S.sha(Path(__file__)),
        'files': {str(q.relative_to(root)): S.sha(q) for q in sorted(root.rglob('*'))
                  if q.is_file() and q != root / 'manifest.json'}})


if __name__ == '__main__':
    publish() if '--publish' in sys.argv else evidence()
