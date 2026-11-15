"""Synthetic-only selector, sparse-regular tie, corruption and binding checks."""
import json
import os
import time
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import numpy as np
from regularity_gap import select
from regularity_gap_study import ROOT,HERE,OUT,PUBLIC,PROTOCOL,CONDITION,bind,sha,save
from audit_heterogeneous_noise import independent_corruption
from audit_regularity_gap import independent_selection


def literal(counts,n):
    pairs=[(i,j) for i in range(n) for j in range(i+1,n)]
    ordered=sorted(int(x) for x in counts);length=len(ordered)
    median=ordered[length//2] if length%2 else (ordered[length//2-1]+ordered[length//2])/2
    rows=[]
    for cut in sorted(set(ordered)):
        if cut>median:continue
        kept=[pair for pair,c in zip(pairs,counts) if c<=cut]
        if not kept:continue
        degree=[sum(cell in pair for pair in kept) for cell in range(n)]
        # A different exact expression: sum of all pairwise degree differences.
        dispersion=sum((degree[i]-degree[j])**2 for i in range(n) for j in range(i+1,n))
        assert (dispersion==0)==(len(set(degree))==1)
        rows.append(dict(cut=cut,retained=len(kept),degree_min=min(degree),degree_max=max(degree),degree_sum=sum(degree),
                         degree_square_sum=sum(d*d for d in degree),dispersion=dispersion))
    best=sorted(rows,key=lambda r:(r['dispersion'],r['retained'],r['cut']))[0]
    return counts<=best['cut'],dict(n=n,median_count=float(median),candidates=rows,selected=best,
                                  minimum_dispersion_ties=sum(r['dispersion']==best['dispersion'] for r in rows))


def main():
    start=time.monotonic();cpu=time.process_time();path=HERE/'regularity-gap-preflight.json';assert not path.exists()
    rng=np.random.default_rng(381);cases=[];candidate_count=0
    for n in [3,4,5,9,81]:
        length=n*(n-1)//2
        arrays=[np.zeros(length,np.int64),np.full(length,7,np.int64)]
        arrays += [rng.integers(0,14,length,dtype=np.int64) for _ in range(4)]
        for counts in arrays:
            expected,record=literal(counts,n);actual,got=select(counts,n)
            audited,audit_record=independent_selection(counts,n)
            assert np.array_equal(actual,expected)
            assert np.array_equal(audited,expected) and audit_record==record
            for key,value in record.items():assert got[key]==value,(n,key)
            candidate_count+=len(record['candidates']);cases.append(dict(n=n,candidates=len(record['candidates'])))
    n=81;ii,jj=np.triu_indices(n,1)
    cycle=(jj==ii+1)|((ii==0)&(jj==n-1));counts=np.where(cycle,0,100).astype(np.int64)
    sparse,record=select(counts,n);expected,reference=literal(counts,n)
    audited,audit_record=independent_selection(counts,n)
    assert np.array_equal(audited,expected) and audit_record==reference
    assert np.array_equal(sparse,cycle) and np.array_equal(sparse,expected)
    assert record['selected']['degree_min']==record['selected']['degree_max']==2
    assert record['minimum_dispersion_ties']==2 and record['selected']['retained']==81
    assert record['candidates']==reference['candidates']
    # This intentionally retains a sparse regular solution over the complete graph.
    # A future real sparse solution must remain a failure if it misses true edges.
    module=bind()
    assert module.OUT==OUT and module.PUBLIC==PUBLIC and module.PROTOCOL==PROTOCOL and module.CONDITIONS==[CONDITION]
    labels=rng.integers(0,9,(19,81),dtype=np.uint8);free=rng.random((19,81))>.35
    uniforms=1-rng.random((19,81));offsets=rng.integers(1,9,(19,81),dtype=np.int64)
    for hot in [np.arange(18),np.arange(63,81)]:
        actual,cells=module.corrupt(np,labels,free,uniforms,offsets,hot,CONDITION)
        expected,reference_cells=independent_corruption(labels,free,uniforms,offsets,hot,CONDITION)
        assert np.array_equal(actual,expected) and np.array_equal(cells,reference_cells)
        assert ((actual!=labels).sum(1)==9).all() and np.array_equal(actual[~free],labels[~free])
    previous=HERE/'heterogeneous-noise-preflight.json';prior=json.loads(previous.read_text());assert prior['status']=='passed'
    # Inherit only unchanged raw-gradient/corruption components, not the old runner binding.
    for name in ['audit_heterogeneous_noise.py','canonical_pl.py','audit_pooled_source_study.py']:
        file=HERE/name;assert sha(file)==prior['source_sha256'][str(file.relative_to(ROOT))]
    sources=[HERE/name for name in ['regularity_gap_study.py','regularity_gap.py','audit_regularity_gap.py','preflight_regularity_gap.py',
                                  'heterogeneous_noise_study.py','audit_heterogeneous_noise.py']]
    result=dict(status='passed',random_constant_cases=cases,total_candidates_checked=candidate_count,
                sparse_regular_tie=dict(selected_degree=2,selected_edges=81,zero_dispersion_candidates=2,no_minimum_degree_guard=True),
                independent_dispersion_identity=True,independent_corruption_cases=2,wrapper_binding_checked=True,
                independent_outcome_auditor_synthetic_parity=True,
                inherited_numerical_preflight_sha256=sha(previous),source_sha256={str(f.relative_to(ROOT)):sha(f) for f in sources},
                process_cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-start,
                scope='Synthetic-only prefreeze checks; no study heatmaps, sources, counts, or outcomes generated or examined.')
    save(path,result);print(json.dumps(result))


if __name__=='__main__':main()
