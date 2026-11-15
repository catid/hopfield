"""Immutable paired source-only comparison; no solver cohort or GPU work."""
import json
from run import ROOT, HERE, sha


def paths(pilot=False):
    assert not pilot
    return ROOT/'artifacts/research/20261114/ripper/correlated_noise', HERE/'protocol-correlated-noise.json'


def load_protocol(pilot=False):
    out, path = paths(pilot)
    p = json.loads(path.read_text())
    registration = json.loads((out/'input_registration.json').read_text())
    assert registration['study_protocol_sha256'] == sha(path)
    assert registration['input_manifest_sha256'] == sha(out/'input_manifest.json')
    assert p['input_protocol_sha256'] == sha(HERE/'protocol-correlated-noise-inputs.json')
    for name, expected in p['code_sha256'].items():
        assert sha(HERE/name) == expected, name
    for name, expected in p['evidence_sha256'].items():
        assert sha(ROOT/name) == expected, name
    p['inputs'] = str((out/'input_manifest.json').relative_to(ROOT))
    return p
