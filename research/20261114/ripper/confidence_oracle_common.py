"""Immutable registration for a post-hoc oracle mechanism study."""
import json
from run import ROOT, HERE, sha
from confidence_budget_inference_common import load_protocol as load_prior_protocol

OUT = ROOT/'artifacts/research/20261114/ripper/confidence_oracle'
PROTOCOL = HERE/'protocol-confidence-oracle.json'


def load_protocol(require_models=True):
    p = json.loads(PROTOCOL.read_text())
    prior = load_prior_protocol()
    assert p['prior_inference_protocol_sha256'] == sha(HERE/'protocol-confidence-budget-inference.json')
    assert p['seeds'] == prior['seeds'] and p['rows_half_open'] == prior['rows_half_open']
    for name, expected in p['code_sha256'].items():
        assert sha(HERE/name) == expected, name
    for name, expected in p['prior_sha256'].items():
        assert sha(ROOT/name) == expected, name
    assert p['rows_half_open'] == [400000, 405000]
    if require_models:
        audit = json.loads((OUT/'model_audit.json').read_text())
        assert audit['status'] == 'validated' and audit['protocol_sha256'] == sha(PROTOCOL)
        assert audit['auditor_sha256'] == sha(HERE/'audit_confidence_oracle_models.py')
        assert audit['models_manifest_sha256'] == sha(OUT/'models.json')
    return p
