"""Frozen source-only numerical gates followed by fresh H1ak inference."""
import json
from run import ROOT, HERE, sha
from confidence_budget_inference_common import load_protocol as load_prior_protocol

OUT = ROOT/'artifacts/research/20261114/ripper/confidence_strict'
PROTOCOL = HERE/'protocol-confidence-strict.json'


def load_protocol(require_models=True):
    p = json.loads(PROTOCOL.read_text())
    prior = load_prior_protocol()
    assert p['seeds'] == prior['seeds'] == list(range(32))
    assert p['rows_half_open'] == [405000,410000]
    for name, expected in p['code_sha256'].items():
        assert sha(HERE/name) == expected, name
    for name, expected in p['prior_sha256'].items():
        assert sha(ROOT/name) == expected, name
    if require_models:
        audit = json.loads((OUT/'model_audit.json').read_text())
        assert audit['status'] == 'validated' and audit['protocol_sha256'] == sha(PROTOCOL)
        assert audit['auditor_sha256'] == sha(HERE/'audit_confidence_strict_models.py')
        assert audit['models_manifest_sha256'] == sha(OUT/'models.json')
    return p
