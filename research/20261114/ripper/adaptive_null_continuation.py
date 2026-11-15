"""Separate additional-budget authorization; the original gate stays failed."""
import json
from run import ROOT,HERE,sha
from adaptive_null_common import load_protocol
PROTOCOL=HERE/'protocol-adaptive-null-continuation.json'

def verify():
    scientific=load_protocol();p=json.loads(PROTOCOL.read_text())
    for name,digest in p['code_sha256'].items():assert sha(HERE/name)==digest,name
    for path,digest in p['input_sha256'].items():assert sha(ROOT/path)==digest,path
    original=json.loads((HERE/'adaptive-null-original-artifacts.json').read_text())
    for path,digest in original['artifact_sha256'].items():assert sha(ROOT/path)==digest,path
    pre=ROOT/'artifacts/research/20261114/ripper/adaptive_null_preflight';runtime=json.loads((pre/'runtime.json').read_text());checks=json.loads((pre/'checks.json').read_text())
    assert runtime['status']=='validated'and checks['status']=='passed'and runtime['checks_sha256']==sha(pre/'checks.json')
    assert checks['corrected_auditor_sha256']==sha(HERE/'audit_adaptive_null_eager.py')and checks['complete_corrected_bank']==0
    assert checks['corrected_bank_audit_sha256']==sha(ROOT/'artifacts/research/20261114/ripper/adaptive_null/s0/audit.json')
    assert p['original_protocol_sha256']==sha(HERE/'protocol-adaptive-null.json')
    return p,scientific,runtime
