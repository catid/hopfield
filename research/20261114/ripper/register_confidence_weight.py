"""Bind all source inputs and execution wrappers before any query or fit."""
import argparse
import json
from run import ROOT,HERE,sha,save,now
from confidence_weight_common import paths


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');args=ap.parse_args()
    out,path=paths(args.pilot);p=json.loads(path.read_text());dest=out/'input_registration.json'
    assert not dest.exists() and not any(out.glob('s*/receipt.json')) and not any(out.glob('s*/calibration.json'))
    m=json.loads((out/'input_manifest.json').read_text());assert len(m['inputs'])==len(p['seeds'])
    if args.pilot:assert m['protocol_sha256']==sha(path)
    else:assert m['protocol_sha256']==p['input_protocol_sha256']==sha(HERE/'protocol-confidence-weight-inputs.json')
    for name,key in [('confidence_weight_policy.py','policy_sha256'),('confidence_weight_estimator.py','estimator_sha256'),('CONFIDENCE_WEIGHTING.md','proof_sha256'),('calibrated_policy.py','base_policy_sha256'),('calibration_estimator.py','base_estimator_sha256')]:assert sha(HERE/name)==p[key]
    pre=ROOT/(p['policy_preflight_path'])
    assert sha(pre)==p['policy_preflight_sha256'];assert json.loads(pre.read_text())['status']=='validated'
    assert sha(ROOT/p['crossing_preflight_path'])==p['crossing_preflight_sha256']
    assert sha(ROOT/p['replay_optimization_validation_path'])==p['replay_optimization_validation_sha256']
    assert sha(HERE/'fast_reference_mask.py')==p['fast_reference_mask_sha256']
    for e in m['inputs']:
        for role in ['training','calibration']:assert sha(ROOT/e[role+'_path'])==e[role+'_sha256']
    r={'status':'registered_before_queries_or_fits','utc':now(),'study_protocol_sha256':sha(path),'input_manifest_sha256':sha(out/'input_manifest.json'),
       'input_protocol_sha256':None if args.pilot else p['input_protocol_sha256'],'registration_sha256':sha(__file__)}
    for name,key in [('confidence_weight_common.py','common_sha256'),('confidence_weight_acquire.py','acquisition_sha256'),('confidence_weight_search.py','search_sha256'),('confidence_weight_policy.py','policy_sha256'),('confidence_weight_estimator.py','estimator_sha256'),('CONFIDENCE_WEIGHTING.md','proof_sha256'),('calibrated_policy.py','base_policy_sha256'),('calibration_estimator.py','base_estimator_sha256')]:r[key]=sha(HERE/name)
    assert sha(HERE/'confidence_integer_audit.py')==p['integer_audit_sha256']
    r['integer_audit_sha256']=p['integer_audit_sha256']
    for item in p['gpu_preflights']:assert sha(ROOT/item['path'])==item['sha256']
    save(dest,r);print(json.dumps(r),flush=True)


if __name__=='__main__':main()
