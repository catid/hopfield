"""Frozen histogram-preserving corruption study on archived H1an source banks."""
import json
from run import ROOT, HERE, sha


def paths():
    return ROOT/'artifacts/research/20261114/ripper/permuted_noise', HERE/'protocol-permuted-noise.json'


def load_protocol(require_registration=True):
    out,path=paths();p=json.loads(path.read_text())
    for name,digest in p['code_sha256'].items():assert sha(HERE/name)==digest,name
    for name,digest in p['prior_sha256'].items():assert sha(ROOT/name)==digest,name
    if require_registration:
        registration=json.loads((out/'input_registration.json').read_text())
        assert registration['protocol_sha256']==sha(path)
        assert registration['preparer_sha256']==sha(HERE/'prepare_permuted_noise.py')
    return p
