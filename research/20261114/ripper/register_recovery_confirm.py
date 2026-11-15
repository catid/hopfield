"""Register input identity before H1x acquisition without editing its protocol."""
import json
from run import ROOT,HERE,sha,save,now


def main():
    path=HERE/'protocol-recovery-confirm.json';p=json.loads(path.read_text());out=ROOT/p['input_registration']
    assert not out.exists() and not any(out.parent.glob('s*/receipt.json'))
    manifest=json.loads((ROOT/p['inputs']).read_text())
    assert manifest['protocol_sha256']==p['input_protocol_sha256']==sha(HERE/'protocol-recovery-confirm-inputs.json')
    assert manifest['total_bank_rows']==153600 and len(manifest['inputs'])==256
    r={'status':'registered_before_acquisition','utc':now(),'study_protocol_sha256':sha(path),
        'input_protocol_sha256':p['input_protocol_sha256'],'input_manifest_sha256':sha(ROOT/p['inputs']),
        'wrapper_sha256':sha(__file__),'common_sha256':sha(HERE/'recovery_confirm_common.py'),
        'acquisition_sha256':sha(HERE/'recovery_confirm_acquire.py'),
        'frozen_policy_sha256':sha(HERE/'exponential_policy.py')}
    save(out,r);print(json.dumps(r))


if __name__=='__main__':main()
