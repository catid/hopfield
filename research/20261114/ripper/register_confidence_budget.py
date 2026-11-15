"""Bind source inputs and every source-only wrapper before calibration or queries."""
import json
from run import ROOT, HERE, sha, save, now
from confidence_budget_common import paths


def main():
    out, path = paths()
    p = json.loads(path.read_text())
    dest = out/'input_registration.json'
    assert not dest.exists() and not any(out.glob('s*/receipt.json')) and not any(out.glob('s*/calibration.json'))
    manifest = json.loads((out/'input_manifest.json').read_text())
    assert manifest['protocol_sha256'] == p['input_protocol_sha256'] == sha(HERE/'protocol-confidence-budget-inputs.json')
    assert len(manifest['inputs']) == len(p['seeds']) == 32
    for name, expected in p['code_sha256'].items():
        assert sha(HERE/name) == expected, name
    for name, expected in p['evidence_sha256'].items():
        assert sha(ROOT/name) == expected, name
    for entry in manifest['inputs']:
        for role in ('training', 'calibration'):
            assert sha(ROOT/entry[role+'_path']) == entry[role+'_sha256']
    r = {'status': 'registered_before_queries_or_fits', 'utc': now(),
         'study_protocol_sha256': sha(path), 'input_protocol_sha256': p['input_protocol_sha256'],
         'input_manifest_sha256': sha(out/'input_manifest.json'), 'registration_sha256': sha(__file__),
         'heldout_access': 'No held-out data or outcomes in this stage', 'code_sha256': p['code_sha256']}
    save(dest, r)
    print(json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
