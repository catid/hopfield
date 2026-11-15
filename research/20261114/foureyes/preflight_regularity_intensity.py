"""H4ag synthetic gate before any new scientific corruption draws."""
import json,os,time
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import numpy as np
from regularity_intensity_study import ROOT,HERE,OUT,PUBLIC,PROTOCOL,CONDITIONS,GROUPS,ROWS,bind,corrupt,sha,save
from audit_regularity_intensity import literal_corruption
from regularity_gap import select
from audit_regularity_gap import independent_selection


def main():
    start=time.monotonic();cpu=time.process_time();path=HERE/'regularity-intensity-preflight.json';assert not path.exists()
    prior=json.loads((HERE/'regularity-fresh-preflight.json').read_text());assert prior['status']=='passed'
    for name in ['regularity_gap.py','audit_regularity_gap.py','heterogeneous_noise_study.py','audit_heterogeneous_noise.py']:
        file=HERE/name;assert sha(file)==prior['source_sha256'][str(file.relative_to(ROOT))]
    module=bind();assert module.OUT==OUT and module.PUBLIC==PUBLIC and module.PROTOCOL==PROTOCOL and module.CONDITIONS==CONDITIONS
    assert GROUPS==64 and ROWS==2400
    rng=np.random.default_rng(822);clean=rng.integers(0,9,(31,81),dtype=np.uint8);free=rng.random((31,81))>.35
    offsets=rng.integers(1,9,(31,81),dtype=np.int64);cases=0;different=0
    for uniforms in [1-rng.random((31,81)),np.ones((31,81)),np.full((31,81),np.nextafter(0.,1.))]:
        for hot in [np.arange(18),np.arange(63,81)]:
            masks=[]
            for condition in CONDITIONS:
                actual,cells=corrupt(np,clean,free,uniforms,offsets,hot,condition)
                reference,chosen=literal_corruption(clean,free,uniforms,offsets,hot,condition)
                assert np.array_equal(actual,reference) and np.array_equal(cells,chosen)
                changed=actual!=clean;assert (changed.sum(1)==9).all() and not changed[~free].any();masks.append(changed)
                if condition=='hot18_weight8':
                    inherited,inherited_cells=module.corrupt(np,clean,free,uniforms,offsets,hot,condition)
                    assert np.array_equal(inherited,actual) and np.array_equal(inherited_cells,cells)
                cases+=1
            assert (masks[1][:,hot].sum(1)>=masks[0][:,hot].sum(1)).all()
            different+=int(np.any(masks[0]!=masks[1],axis=1).sum())
    assert different>0
    candidates=0
    for counts in [np.zeros(3240,np.int64),rng.integers(0,2401,3240,dtype=np.int64)]:
        support,record=select(counts);independent,reference=independent_selection(counts)
        assert np.array_equal(support,independent)
        for key,value in reference.items():assert record[key]==value
        candidates+=len(reference['candidates'])
    files=[HERE/name for name in ['regularity_intensity_study.py','audit_regularity_intensity.py','preflight_regularity_intensity.py',
                                'regularity_gap.py','audit_regularity_gap.py','heterogeneous_noise_study.py','audit_heterogeneous_noise.py']]
    result=dict(status='passed',independent_literal_corruption_cases=cases,different_masks=different,
                boundary_cases='U=1 and smallest positive float; stable cell ties, nonzero digit offsets, clues preserved',
                monotone_hot_selection=True,inherited_weight8_parity=True,threshold_candidates_checked=candidates,worker_binding_checked=True,
                source_sha256={str(f.relative_to(ROOT)):sha(f) for f in files},wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu)
    save(path,result);print(json.dumps(result))


if __name__=='__main__':main()
