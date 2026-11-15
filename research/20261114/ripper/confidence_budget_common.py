"""Frozen source-only H1ai configuration; held-out inference is a separate stage."""
import json
from run import ROOT, HERE, sha


def paths(pilot=False):
    assert not pilot, 'H1ai has no outcome pilot and no held-out inference in this stage'
    return ROOT/'artifacts/research/20261114/ripper/confidence_budget', HERE/'protocol-confidence-budget-source.json'


def load_protocol(pilot=False):
    out, path = paths(pilot)
    p = json.loads(path.read_text())
    r = json.loads((out/'input_registration.json').read_text())
    assert r['study_protocol_sha256'] == sha(path)
    assert r['input_manifest_sha256'] == sha(out/'input_manifest.json')
    assert p['input_protocol_sha256'] == sha(HERE/'protocol-confidence-budget-inputs.json')
    for name, expected in p['code_sha256'].items():
        assert sha(HERE/name) == expected, name
    for name, expected in p['evidence_sha256'].items():
        assert sha(ROOT/name) == expected, name
    p['inputs'] = str((out/'input_manifest.json').relative_to(ROOT))
    p['input_manifest_sha256'] = r['input_manifest_sha256']
    return p
