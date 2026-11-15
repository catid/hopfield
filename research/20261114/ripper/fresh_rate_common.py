"""Frozen H1az paths and source audit binding for fresh acquisition."""
import json
from run import ROOT,HERE,sha


def paths():return ROOT/'artifacts/research/20261114/ripper/fresh_rate',HERE/'protocol-fresh-rate.json'


def load_protocol(require_registration=True):
    out,path=paths();p=json.loads(path.read_text())
    for name,digest in p['code_sha256'].items():assert sha(ROOT/name)==digest,name
    if require_registration:
        sealed=json.loads((out/'identity-seal.json').read_text());audit=json.loads((out/'source-audit.json').read_text())
        assert sealed['protocol_sha256']==audit['protocol_sha256']==sha(path) and audit['status']=='validated'
        assert sealed['selected_sources_sha256']==sha(out/'selected_sources.npz')
        assert audit['input_manifest_sha256']==sha(out/'input_manifest.json') and audit['identity_seal_sha256']==sha(out/'identity-seal.json')
    return p
