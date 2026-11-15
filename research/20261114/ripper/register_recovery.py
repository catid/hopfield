"""Bind fresh inputs to the already-frozen H1w protocol before acquisition."""
import json
from fractions import Fraction
import numpy as np
from run import ROOT,HERE,sha,save,now
from exponential_policy import predictive_weights


def main():
    path=HERE/'protocol-recovery.json';p=json.loads(path.read_text());out=ROOT/p['input_registration']
    assert not out.exists() and not any(out.parent.glob('s*/receipt.json'))
    manifest=json.loads((ROOT/p['inputs']).read_text())
    assert manifest['protocol_sha256']==p['input_protocol_sha256']==sha(HERE/'protocol-recovery-inputs.json')
    assert manifest['fresh_unique_rows']==63454 and manifest['total_bank_rows']==64000
    weights=[];zero=np.zeros((81,81),np.int32)
    for m in range(1001):
        w=predictive_weights(np.full((81,81),m,np.int32),zero)
        assert np.isclose(w[0,1],float(Fraction(8,9)**m),rtol=1e-13,atol=0)
        weights.append(w[0,1])
    assert weights[0]==1 and np.all(np.diff(weights)<0) and min(weights)>0
    r={'status':'registered_before_acquisition','utc':now(),'study_protocol_sha256':sha(path),
        'input_protocol_sha256':p['input_protocol_sha256'],'input_manifest_sha256':sha(ROOT/p['inputs']),
        'wrapper_sha256':sha(__file__),'extended_policy_formula_checks':1001,'formula_relative_tolerance':1e-13,
        'clarification_sha256':sha(HERE/'RECOVERY_PROTOCOL_CLARIFICATION.md'),
        'common_sha256':sha(HERE/'recovery_common.py'),'acquisition_sha256':sha(HERE/'recovery_acquire.py'),
        'search_sha256':sha(HERE/'recovery_search.py'),'frozen_policy_sha256':sha(HERE/'exponential_policy.py')}
    save(out,r);print(json.dumps(r))


if __name__=='__main__':main()
