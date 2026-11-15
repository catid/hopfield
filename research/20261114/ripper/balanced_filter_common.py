"""Frozen paired acquisition with an explicitly supplied digit-balance prior."""
import json
from run import ROOT,HERE,sha


def paths(pilot=False):
    assert not pilot
    return ROOT/'artifacts/research/20261114/ripper/balanced_filter', HERE/'protocol-balanced-filter.json'


def load_protocol(pilot=False):
    out,path=paths(pilot);p=json.loads(path.read_text())
    registration=json.loads((out/'input_registration.json').read_text())
    assert registration['study_protocol_sha256']==sha(path)
    assert registration['input_manifest_sha256']==sha(out/'input_manifest.json')
    assert p['input_protocol_sha256']==sha(HERE/'protocol-balanced-filter-inputs.json')
    for name,digest in p['code_sha256'].items():assert sha(HERE/name)==digest,name
    for name,digest in p['evidence_sha256'].items():assert sha(ROOT/name)==digest,name
    p['inputs']=str((out/'input_manifest.json').relative_to(ROOT))
    return p
