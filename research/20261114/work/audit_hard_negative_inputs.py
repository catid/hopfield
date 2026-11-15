"""Independent H2ac shortest labels, uniqueness and previous-cohort disjointness."""
from concurrent.futures import ProcessPoolExecutor
import json
import time
import numpy as np
import study as S
import audit as A
from audit_larger_relaxation import check_dataset
from audit_contrastive_inputs import digest
from hard_negative_transfer import PP,DEST


def main():
    started=time.perf_counter();manifest=json.loads((DEST/'datasets.json').read_text());assert manifest['protocol_sha256']==S.sha(PP);fresh=set()
    for ds in manifest['datasets']:
        assert S.sha(S.ROOT/ds['path'])==ds['sha256']
        with np.load(S.ROOT/ds['path']) as x:fresh.update(digest(t) for t in x['tokens'])
    assert len(fresh)==4096
    with ProcessPoolExecutor(max_workers=8) as pool:distances=list(pool.map(check_dataset,[S.ROOT/ds['path'] for ds in manifest['datasets']]))
    prior=[]
    for cohort in ['native_relaxation','larger_relaxation','selective_fallback','window_fallback','contrastive_training','lp_transfer','lp_relaxation','binary_coordinate']:
        mp=S.OUT/cohort/'datasets.json';old=json.loads(mp.read_text());n=overlap=0
        for ds in old['datasets']:
            with np.load(S.ROOT/ds['path']) as x:
                for tok in x['tokens']:n+=1;overlap+=digest(tok) in fresh
        assert overlap==0;prior.append({'cohort':cohort,'n':n,'overlap':overlap,'manifest_sha256':S.sha(mp)})
    result={'status':'passed','utc':S.utc(),'protocol_sha256':S.sha(PP),'datasets':manifest['datasets'],'distances':distances,'inputs_checked':4096,'unique':4096,'prior_cohort_checks':prior,'processes':8,'seconds':time.perf_counter()-started,'script_sha256':S.sha(__file__),'graph_checker_sha256':S.sha(A.__file__),'outcome_data_read':False};S.save(DEST/'independent-input-audit.json',result);S.emit('hard_negative_inputs_audited',inputs=4096,seconds=result['seconds'],overlaps=0)


if __name__=='__main__':main()
