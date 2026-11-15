"""H4af synthetic sampling, corruption, frozen selector and worker binding gate."""
import json
import os
import time
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import numpy as np
from regularity_fresh_study import ROOT,HERE,OUT,PUBLIC,PROTOCOL,CONDITION,GROUPS,ROWS,bind,sha,save
from audit_heterogeneous_noise import independent_corruption
from regularity_gap import select
from audit_regularity_gap import independent_selection


def main():
    start=time.monotonic();cpu=time.process_time();path=HERE/'regularity-fresh-preflight.json';assert not path.exists()
    prior=json.loads((HERE/'regularity-gap-preflight.json').read_text());assert prior['status']=='passed'
    for name in ['regularity_gap.py','audit_regularity_gap.py','heterogeneous_noise_study.py','audit_heterogeneous_noise.py']:
        file=HERE/name;assert sha(file)==prior['source_sha256'][str(file.relative_to(ROOT))]
    module=bind();assert module.OUT==OUT and module.PUBLIC==PUBLIC and module.PROTOCOL==PROTOCOL and module.CONDITIONS==[CONDITION]
    assert GROUPS==64 and ROWS==2400
    rng=np.random.default_rng(477);fake_excluded=np.unique(rng.integers(0,10000,2700));fake_eligible=np.setdiff1d(np.arange(10000),fake_excluded)
    excluded_set=set(fake_excluded)
    assert np.array_equal(fake_eligible,np.asarray([r for r in range(10000) if r not in excluded_set]))
    draws=np.stack([np.random.default_rng(37000+g).choice(fake_eligible,ROWS,replace=False) for g in range(GROUPS)])
    assert all(len(np.unique(row))==ROWS for row in draws) and not np.isin(draws,fake_excluded).any()
    assert draws.shape==(64,2400) and len(np.unique(draws))<draws.size
    labels=rng.integers(0,9,(31,81),dtype=np.uint8);free=rng.random((31,81))>.35
    uniforms=1-rng.random((31,81));offsets=rng.integers(1,9,(31,81),dtype=np.int64)
    for hot in [np.arange(18),np.arange(63,81)]:
        actual,cells=module.corrupt(np,labels,free,uniforms,offsets,hot,CONDITION)
        expected,reference=independent_corruption(labels,free,uniforms,offsets,hot,CONDITION)
        assert np.array_equal(actual,expected) and np.array_equal(cells,reference)
        assert ((actual!=labels).sum(1)==9).all() and np.array_equal(actual[~free],labels[~free])
    candidates=0
    for counts in [np.zeros(3240,np.int64),rng.integers(0,2401,3240,dtype=np.int64)]:
        support,record=select(counts);independent,reference=independent_selection(counts)
        assert np.array_equal(support,independent)
        for key,value in reference.items():assert record[key]==value
        candidates+=len(reference['candidates'])
    files=[HERE/name for name in ['regularity_fresh_study.py','audit_regularity_fresh.py','preflight_regularity_fresh.py',
                                'regularity_gap.py','audit_regularity_gap.py','heterogeneous_noise_study.py','audit_heterogeneous_noise.py']]
    result=dict(status='passed',synthetic_banks=64,synthetic_rows_per_bank=2400,synthetic_exclusion_and_withinbank_uniqueness=True,
                crossbank_overlap_retained=True,independent_corruption_cases=2,threshold_candidates_checked=candidates,
                worker_binding_checked=True,inherited_selector_preflight_sha256=sha(HERE/'regularity-gap-preflight.json'),
                source_sha256={str(f.relative_to(ROOT)):sha(f) for f in files},
                process_cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-start,
                scope='Synthetic-only gate; no fresh study source IDs, heatmaps, corruptions, counts or outcomes drawn or inspected.')
    save(path,result);print(json.dumps(result))


if __name__=='__main__':main()
