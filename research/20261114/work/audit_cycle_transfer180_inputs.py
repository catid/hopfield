"""H2av fresh-label/uniqueness/nonoverlap check against frozen complete registry."""
from concurrent.futures import ProcessPoolExecutor
import json
import time
import numpy as np
import study as S
from audit_larger_relaxation import check_dataset
from audit_contrastive_inputs import digest
from cycle_transfer_180 import PP,DEST


def main():
    started=time.perf_counter();p=json.loads(PP.read_text());manifest=json.loads((DEST/'datasets.json').read_text());assert manifest['protocol_sha256']==S.sha(PP);fresh=set()
    for ds in manifest['datasets']:
        assert S.sha(S.ROOT/ds['path'])==ds['sha256']
        with np.load(S.ROOT/ds['path']) as z:fresh.update(digest(t) for t in z['tokens'])
    assert len(fresh)==4096
    with ProcessPoolExecutor(max_workers=8) as pool:distances=list(pool.map(check_dataset,[S.ROOT/ds['path'] for ds in manifest['datasets']]))
    prior=[];seen_files=set()
    for registered in p['cohort']['prior_registry']:
        path=S.ROOT/registered['path'];assert S.sha(path)==registered['sha256'];old=json.loads(path.read_text());count=overlap=0;unique_files=0
        datasets=old['datasets'].values() if isinstance(old['datasets'],dict) else old['datasets']
        for ds in datasets:
            fp=S.ROOT/ds['path']
            if fp in seen_files:continue
            seen_files.add(fp);unique_files+=1
            with np.load(fp) as z:
                for tok in z['tokens']:count+=1;overlap+=digest(tok) in fresh
        assert overlap==0;prior.append({**registered,'newly_checked_files':unique_files,'inputs_checked':count,'overlap':overlap})
    r={'status':'passed','protocol_sha256':S.sha(PP),'datasets':manifest['datasets'],'distances':distances,'inputs_checked':4096,'unique':4096,'prior_manifest_checks':prior,'prior_unique_dataset_files':len(seen_files),'processes':8,'seconds':time.perf_counter()-started,'source_sha256':S.sha(__file__),'graph_checker_sha256':S.sha(S.HERE/'audit_larger_relaxation.py'),'outcome_data_read':False};assert r['seconds']<120;S.save(DEST/'independent-input-audit.json',r);S.emit('cycle_transfer_inputs_audited',seconds=r['seconds'],prior_files=len(seen_files),overlaps=0)


if __name__=='__main__':main()
