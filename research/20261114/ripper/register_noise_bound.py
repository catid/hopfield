"""Register source hashes and unchanged noisy policy before fresh acquisition."""
import json
from run import ROOT,HERE,sha,save,now


def main():
    path=HERE/'protocol-noise-bound.json';p=json.loads(path.read_text());out=ROOT/p['input_registration']
    assert not out.exists() and not any(out.parent.glob('s*/receipt.json'))
    manifest=json.loads((ROOT/p['inputs']).read_text())
    assert manifest['protocol_sha256']==p['input_protocol_sha256']==sha(HERE/'protocol-noise-bound-inputs.json')
    assert manifest['total_bank_rows']==48000 and len(manifest['inputs'])==32
    assert p['policy_sha256']==sha(HERE/'noise_bound_policy.py') and p['proof_sha256']==sha(HERE/'NOISE_UPPER_BOUND_EPROCESS.md')
    assert p['original_policy_sha256']==sha(HERE/'online_policy.py')
    preflight=json.loads((ROOT/p['policy_preflight_path']).read_text())
    assert sha(ROOT/p['policy_preflight_path'])==p['policy_preflight_sha256'] and preflight['status']=='validated'
    assert preflight['policy_sha256']==p['policy_sha256'] and preflight['exact_control_thresholds_unchanged']
    assert preflight['conditional_null_expectations']['exact']['value']==1
    assert preflight['conditional_null_expectations']['upper']['value']<1<preflight['conditional_null_expectations']['under']['value']
    assert p['original_noisy_policy_sha256']==sha(HERE/'noisy_policy.py')
    r={'status':'registered_before_acquisition','utc':now(),'study_protocol_sha256':sha(path),
        'input_protocol_sha256':p['input_protocol_sha256'],'input_manifest_sha256':sha(ROOT/p['inputs']),
        'wrapper_sha256':sha(__file__),'common_sha256':sha(HERE/'noise_bound_common.py'),
        'acquisition_sha256':sha(HERE/'noise_bound_acquire.py'),'search_sha256':sha(HERE/'noise_bound_search.py'),
        'policy_sha256':p['policy_sha256'],'proof_sha256':p['proof_sha256']}
    save(out,r);print(json.dumps(r))


if __name__=='__main__':main()
