"""Frozen H1ai inference stage over every independently audited source bank."""
import json
from run import ROOT, HERE, sha
from confidence_budget_common import load_protocol as load_source_protocol


def load_protocol():
    p = json.loads((HERE/'protocol-confidence-budget-inference.json').read_text())
    source = load_source_protocol()
    out = ROOT/'artifacts/research/20261114/ripper/confidence_budget'
    assert p['source_protocol_sha256'] == sha(HERE/'protocol-confidence-budget-source.json')
    assert p['source_audit_sha256'] == sha(out/'source_audit.json')
    audit = json.loads((out/'source_audit.json').read_text())
    assert audit['status'] == 'validated' and audit['source_only'] and audit['heldout_inputs_accessed'] == 0
    assert p['seeds'] == source['seeds'] == list(range(32))
    assert p['arms'] == source['arms'] == ['uniform', 'evidence', 'shuffled']
    assert p['streams'] == source['streams']
    assert p['input_manifest_sha256'] == sha(out/'input_manifest.json')
    for name, expected in p['code_sha256'].items():
        assert sha(HERE/name) == expected, name
    for item in p['reused_gpu_preflights']:
        assert sha(ROOT/item['path']) == item['sha256']
        assert json.loads((ROOT/item['path']).read_text())['status'] == 'validated'
    assert p['gpu_preflight_seconds'] == 0
    assert p['rows_half_open'] == [400000, 405000]
    return p
