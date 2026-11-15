"""Resolve the post-preparation input hash without changing the frozen study."""
import json
from run import ROOT,HERE,sha


def load_protocol():
    path=HERE/'protocol-recovery-confirm.json';p=json.loads(path.read_text())
    r=json.loads((ROOT/p['input_registration']).read_text())
    assert r['study_protocol_sha256']==sha(path)
    assert r['input_protocol_sha256']==p['input_protocol_sha256']==sha(HERE/'protocol-recovery-confirm-inputs.json')
    assert r['input_manifest_sha256']==sha(ROOT/p['inputs'])
    assert p['input_manifest_sha256'] is None
    p['input_manifest_sha256']=r['input_manifest_sha256']
    return p
