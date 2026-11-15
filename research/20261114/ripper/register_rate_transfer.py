"""Bind all source inputs and execution wrappers before any query or fit."""
import argparse
import json
from run import ROOT,HERE,sha,save,now
from rate_transfer_common import paths


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');args=ap.parse_args()
    out,path=paths(args.pilot);p=json.loads(path.read_text());dest=out/'input_registration.json'
    assert not dest.exists() and not any(out.glob('s*/receipt.json')) and not any(out.glob('s*/calibration.json'))
    m=json.loads((out/'input_manifest.json').read_text());assert len(m['inputs'])==len(p['seeds'])
    if args.pilot:assert m['protocol_sha256']==sha(path)
    else:assert m['protocol_sha256']==p['input_protocol_sha256']==sha(HERE/'protocol-rate-transfer-inputs.json')
    for name,key in [('rate_transfer_policy.py','policy_sha256'),('rate_transfer_estimator.py','estimator_sha256'),('RATE_TRANSFER.md','proof_sha256'),('calibrated_policy.py','base_policy_sha256'),('calibration_estimator.py','base_estimator_sha256')]:assert sha(HERE/name)==p[key]
    pre=ROOT/(p['policy_preflight_path'])
    assert sha(pre)==p['policy_preflight_sha256'];assert json.loads(pre.read_text())['status']=='validated'
    for e in m['inputs']:
        for role in ['training','calibration']:assert sha(ROOT/e[role+'_path'])==e[role+'_sha256']
    r={'status':'registered_before_queries_or_fits','utc':now(),'study_protocol_sha256':sha(path),'input_manifest_sha256':sha(out/'input_manifest.json'),
       'input_protocol_sha256':None if args.pilot else p['input_protocol_sha256'],'registration_sha256':sha(__file__)}
    for name,key in [('rate_transfer_common.py','common_sha256'),('rate_transfer_acquire.py','acquisition_sha256'),('rate_transfer_search.py','search_sha256'),('rate_transfer_policy.py','policy_sha256'),('rate_transfer_estimator.py','estimator_sha256'),('RATE_TRANSFER.md','proof_sha256'),('calibrated_policy.py','base_policy_sha256'),('calibration_estimator.py','base_estimator_sha256')]:r[key]=sha(HERE/name)
    save(dest,r);print(json.dumps(r),flush=True)


if __name__=='__main__':main()
