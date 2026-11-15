"""Synthetic correctness and archived audit parallelism gate, before H4ak draws."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import json,time,concurrent.futures,ast
from pathlib import Path
import numpy as np
from fresh_curvature_study import ROOT,HERE,OUT,PUBLIC,PROTOCOL,CONDITION,bind,corrupt,sha,save
from audit_regularity_intensity import literal_corruption
from regularity_gap import select
from audit_regularity_gap import independent_selection


def archived_bank(index):
    from audit_degree_curvature import bank,npz
    folder=ROOT/'results/research/foureyes/degree_curvature'/f'j{index:03d}'
    data=npz(folder/'models.npz');receipt=json.loads((folder/'receipt.json').read_text());p=json.loads((HERE/'degree_curvature.json').read_text())
    return bank(index,data,receipt,data['cost'],p)


def main():
    path=HERE/'fresh-curvature-preflight.json';assert not path.exists() and not PROTOCOL.exists() and not OUT.exists()
    start=time.monotonic();module=bind();assert module.OUT==OUT and module.PUBLIC==PUBLIC and module.PROTOCOL==PROTOCOL and module.CONDITIONS==[CONDITION]
    rng=np.random.default_rng(4881);excluded=np.unique(rng.integers(0,10000,2700));eligible=np.setdiff1d(np.arange(10000),excluded)
    exclusion=set(excluded);assert np.array_equal(eligible,[i for i in range(10000) if i not in exclusion])
    draws=np.stack([np.random.default_rng(38000+g).choice(eligible,2400,replace=False) for g in range(64)])
    assert all(len(np.unique(row))==2400 for row in draws) and not np.isin(draws,excluded).any()
    labels=rng.integers(0,9,(31,81),dtype=np.uint8);free=rng.random((31,81))>.35;uniforms=1-rng.random((31,81));offsets=rng.integers(1,9,(31,81),dtype=np.int64)
    for hot in [np.arange(18),np.arange(63,81)]:
        actual,cells=corrupt(np,labels,free,uniforms,offsets,hot,CONDITION);expected,ref=literal_corruption(labels,free,uniforms,offsets,hot,CONDITION)
        assert np.array_equal(actual,expected) and np.array_equal(cells,ref)
        assert ((actual!=labels).sum(1)==9).all() and np.array_equal(actual[~free],labels[~free])
    for counts in [np.zeros(3240,np.int64),rng.integers(0,2401,3240,dtype=np.int64)]:
        a,r=select(counts);b,t=independent_selection(counts);assert np.array_equal(a,b)
        for k,v in t.items():assert r[k]==v
    begin=time.monotonic();serial=list(map(archived_bank,range(8)));serial_s=time.monotonic()-begin
    begin=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:parallel=list(pool.map(archived_bank,range(8)))
    parallel_s=time.monotonic()-begin;assert serial==parallel
    dependencies=set()
    for name in ['regularity_fresh.json','regularity_intensity.json','degree_curvature.json','centered_regularity.json']:
        prior=json.loads((HERE/name).read_text())
        for rel,digest in prior['source_sha256'].items():assert sha(ROOT/rel)==digest;dependencies.add(rel)
    dependencies.update(str((HERE/name).relative_to(ROOT)) for name in ['fresh_curvature_study.py','audit_fresh_curvature_controls.py','fresh_curvature_degree.py','audit_fresh_curvature_degree.py','preflight_fresh_curvature.py','audit_regularity_intensity.py','audit_centered_regularity.py'])
    for rel in dependencies:ast.parse((ROOT/rel).read_text())
    result=dict(status='passed',no_fresh_study_draws=True,synthetic_banks=64,synthetic_rows_per_bank=2400,corruption_cases=2,worker_binding=True,
                archived_parallel_parity_banks=8,archived_serial_wall_seconds=serial_s,archived_parallel_wall_seconds=parallel_s,archived_subset_speed_ratio=serial_s/parallel_s,
                timing_scope='One archived eight-bank paired audit; overhead included; no general timing guarantee.',
                source_sha256={rel:sha(ROOT/rel) for rel in sorted(dependencies)},wall_seconds=time.monotonic()-start)
    save(path,result);print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))


if __name__=='__main__':main()
