"""Immutable registration and shared pilot/fresh paths."""
import json
from run import ROOT,HERE,sha


def paths(pilot=False):
    out=ROOT/('artifacts/research/20261114/ripper/scaled_noise_pilot' if pilot else 'artifacts/research/20261114/ripper/scaled_noise')
    protocol=HERE/('protocol-scaled-noise-pilot.json' if pilot else 'protocol-scaled-noise.json')
    return out,protocol


def load_protocol(pilot=False):
    out,path=paths(pilot);p=json.loads(path.read_text());r=json.loads((out/'input_registration.json').read_text())
    assert r['study_protocol_sha256']==sha(path)
    assert r['input_manifest_sha256']==sha(out/'input_manifest.json')
    for name,key in [('scaled_noise_common.py','common_sha256'),('scaled_noise_acquire.py','acquisition_sha256'),('scaled_noise_search.py','search_sha256'),('scaled_noise_policy.py','policy_sha256'),('scaled_noise_estimator.py','estimator_sha256'),('SCALED_NOISE.md','proof_sha256'),('calibrated_policy.py','base_policy_sha256'),('calibration_estimator.py','base_estimator_sha256')]:
        assert sha(HERE/name)==r[key]
    if not pilot:assert r['input_protocol_sha256']==p['input_protocol_sha256']==sha(HERE/'protocol-scaled-noise-inputs.json')
    p['inputs']=str((out/'input_manifest.json').relative_to(ROOT));p['input_registration']=str((out/'input_registration.json').relative_to(ROOT))
    p['input_manifest_sha256']=r['input_manifest_sha256']
    return p
