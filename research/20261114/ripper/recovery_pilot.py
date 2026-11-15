"""Exploratory fixed-policy recovery curve on six already-used source banks."""
import json
import numpy as np
from run import ROOT,HERE,sha,save,now
from exponential_policy import choose_mask,learn_queried
from online_policy import choose_mask as original


def main():
    assert np.__version__=='2.3.4'
    pp=HERE/'protocol-recovery-pilot.json';p=json.loads(pp.read_text())
    mp=ROOT/p['source_manifest'];assert sha(mp)==p['source_manifest_sha256']
    assert sha(HERE/'exponential_policy.py')==p['policy_sha256']
    manifest=json.loads(mp.read_text());rows=[];i,j=np.triu_indices(81,1)
    peer=(i//9==j//9)|(i%9==j%9)|((i//27==j//27)&(i%9//3==j%9//3))
    for seed in p['seeds']:
        e=next(e for e in manifest['inputs'] if e['seed']==seed)
        path=ROOT/e['completion_path'];assert sha(path)==e['completion_sha256']
        z=np.load(path);rng=np.random.default_rng(20261125+seed)
        states={a:(np.zeros((81,81),np.int32),np.zeros((81,81),np.int32)) for a in p['arms']}
        for n in range(1,1001):
            order=rng.permutation(81)
            for arm,(joint,equal) in states.items():
                function=original if arm=='uniform' else choose_mask
                mask=function(arm,joint,equal,25,order)
                learn_queried(joint,equal,mask,z['solutions'][n-1,mask])
                if n in p['sizes']:
                    support=equal[i,j]==0;missing=int((peer&~support).sum());assert missing==0
                    rows.append({'seed':seed,'arm':arm,'n':n,'extra_edges':int((~peer&support).sum()),'missing_true_edges':missing})
    report={'status':'exploratory','utc':now(),'protocol_sha256':sha(pp),'wrapper_sha256':sha(__file__),'rows':rows,'scope':p['scope']}
    save(ROOT/'artifacts/research/20261114/ripper/recovery_pilot.json',report)
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
