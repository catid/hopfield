"""Synthetic-only thinning, nonzero true-edge costs, proofs and primary boundaries."""
from pathlib import Path
import time
import numpy as np
import partial_mask_recall as S
from partial_mask_math import draws, revealed_mask, independent_draws, independent_mask, threshold
from corruption_censor_math import score, independent_score
from audit_partial_mask_recall import aggregate
from degree_prior_solver import solve
from degree_prior_math import attempt_certificate
from audit_degree_prior import check_certificate


def seed_audit(design):
    values, files = set(), {}
    def scan(value, context=False):
        if isinstance(value,dict):
            for key,item in value.items():
                scan(item,context or 'seed' in key.lower())
        elif isinstance(value,list):
            for item in value:
                scan(item,context)
        elif context and type(value) is int:
            values.add(value)
    for path in sorted((S.ROOT/'research').rglob('*.json')):
        try:
            data=S.read(path)
        except (ValueError,UnicodeDecodeError):
            continue
        if isinstance(data,dict) and data.get('frozen_utc'):
            files[str(path.relative_to(S.ROOT))]=S.sha(path)
            scan(data)
    parent=S.read(S.PARENT_PROTOCOL)
    parent_base=parent['solver']['random_seed_base']
    expanded_parent=set(range(parent_base,parent_base+len(parent['jobs'])))
    proposed=set(range(design['solver_seed_base'],design['solver_seed_base']+1280))|{design['thinning']['seed_base']}
    assert len(proposed)==1281 and not proposed & values
    assert not proposed & expanded_parent and max(proposed)<2**31
    S.save(S.HERE/'partial-mask-recall-seed-audit.json',dict(status='passed',frozen_protocol_sha256=files,
           historical_explicit_seed_values=len(values),solver_seed_range=[design['solver_seed_base'],design['solver_seed_base']+1279],
           thinning_SeedSequence_base=design['thinning']['seed_base'],collisions=[],
           expanded_H4at_solver_seed_range=[min(expanded_parent),max(expanded_parent)],
           scope='Checks explicit integer seed fields in snapshotted protocols plus all256 actual H4at solver seeds. Does not expand every other historical formula or prove stream independence.'))


def fixture(successes,unknown):
    rows=[]
    for i in range(256):
        for p in S.RECALLS:
            kind='success' if i<successes else 'unresolved' if i<successes+unknown else 'resolved_failure_nonexact'
            rows.append(dict(index=i,recall_percent=p,classification=kind,certified=kind!='unresolved',
                             family=S.FAMILIES[i%4],baseline_exact=i<129,revealed_indices=2700))
    return aggregate(rows)


def main():
    assert not S.PROTOCOL.exists() and not S.OUT.exists() and not S.PUBLIC.exists()
    target=S.HERE/'partial-mask-recall-preflight.json'
    assert not target.exists()
    start=time.perf_counter()
    design=S.read(S.HERE/'partial-mask-recall-proposal.json')
    seed_audit(design)
    rng=np.random.default_rng(982050)
    cells=np.array([rng.choice(81,9,replace=False) for _ in range(600)],np.int64)
    noisy=rng.integers(0,9,(600,81),dtype=np.int8)
    raw=draws(982051,7)
    assert np.array_equal(raw,independent_draws(982051,7))
    previous=np.zeros((600,81),bool)
    for p in [0,*S.RECALLS,100]:
        vector=revealed_mask(cells,raw,p)
        scalar=independent_mask(cells,raw,p)
        assert np.array_equal(vector,scalar) and not np.any(previous & ~vector)
        for key,val in score(noisy,vector).items():
            assert np.array_equal(val,independent_score(noisy,scalar)[key])
        if p==0:
            assert not vector.any()
        if p==100:
            assert vector.sum()==5400 and (vector.sum(1)==9).all()
        previous=vector
    perm=rng.permutation(9)
    assert np.array_equal(revealed_mask(cells,raw,90),revealed_mask(cells[:,perm],raw[:,perm],90))
    edges=[]
    for p in [0,*S.RECALLS,100]:
        limit=threshold(p)
        candidates=[0,(1<<64)-1]+[v for v in [limit-1,limit,limit+1] if 0<=v<(1<<64)]
        values=(candidates*9)[:9]
        packet=np.array([values],np.uint64)
        at=revealed_mask(np.arange(9,dtype=np.int64)[None],packet,p)
        assert [bool(at[0,k]) for k in range(9)]==[v<limit for v in values]
        assert np.array_equal(at,independent_mask(np.arange(9,dtype=np.int64)[None],packet,p))
        edges.append(dict(percent=p,threshold=str(limit),boundary_values=list(map(str,values))))
    for args in [(cells,raw.astype(np.int64),90),(cells,raw[:-1],90),(cells.astype(np.int32),raw,90),(cells,raw,-1),(cells,raw,101),(cells,raw,90.)]:
        try:
            revealed_mask(*args)
        except AssertionError:
            pass
        else:
            raise AssertionError('Malformed thinning inputs accepted')
    # A missed actual corruption can create a true-edge collision: preserve it.
    clean=np.array([[0,1,2,3]],np.int8)
    observed=clean.copy();observed[0,0]=1
    complete=np.array([[True,False,False,False]])
    missed=np.zeros((1,4),bool)
    assert score(observed,complete)['cost'][0]==0 and score(observed,missed)['cost'][0]==1
    assert not score(observed,np.ones((1,4),bool))['exposure'].any()
    folder=S.ROOT/'artifacts/research/20261114/foureyes/partial_mask_recall_preflight'
    folder.mkdir(parents=True,exist_ok=True)
    ii,jj=np.triu_indices(4,1)
    cost=np.ones(6,np.int64)
    x,y,meta=solve(4,1,ii,jj,cost,982052-2026111502,folder/'positive.log')
    cert,proof=attempt_certificate(4,1,ii,jj,cost,x,y)
    w=[proof[k] for k in ['primal_twice','node_dual_twice','upper_dual_twice']]
    assert meta['solver_optimal'] and cert['certified'] and cert['primal_objective_twice']==4
    assert check_certificate(4,1,ii,jj,cost,*w)
    for k in range(3):
        bad=[v.copy() for v in w];bad[k][0]+=1
        assert not check_certificate(4,1,ii,jj,cost,*bad)
    endpoints=[]
    for successes,unknown,expected in [(245,0,'supported'),(244,0,'falsified'),(244,1,'inconclusive'),(0,256,'inconclusive'),(256,0,'supported')]:
        result=fixture(successes,unknown)
        assert result['primary']['decision']==expected
        endpoints.append(result['primary'])
    timings={}
    for fn in [revealed_mask,independent_mask]:
        samples=[]
        for _ in range(5):
            t=time.perf_counter();answer=fn(cells,raw,90);samples.append(time.perf_counter()-t)
            assert np.array_equal(answer,revealed_mask(cells,raw,90))
        timings[fn.__name__]=float(np.median(samples))
    sources=S.read(S.PARENT_PROTOCOL)['source_sha256'].copy()
    for name in ['partial_mask_math.py','partial_mask_recall.py','audit_partial_mask_recall.py','preflight_partial_mask_recall.py']:
        path=S.HERE/name;sources[str(path.relative_to(S.ROOT))]=S.sha(path)
    S.save(target,dict(status='passed',study_partial_masks_or_scores_read=False,threshold_boundaries=edges,
           rowwise_vs_batch_uint64_parity=True,nesting_precision_and_index_permutation=True,
           missed_corruption_positive_true_edge_cost_retained=True,zero_exposure_identity=True,
           invalid_inputs_rejected=6,positive_objective_certificate=cert,mutations_rejected=3,
           primary_boundaries=endpoints,mask_timings_seconds=timings,
           performance_scope='Unrelated600x81 fixture only; complete scientific runtime not measured.',
           source_sha256=sources,seed_audit_sha256=S.sha(S.HERE/'partial-mask-recall-seed-audit.json'),
           gate_wall_seconds=time.perf_counter()-start))
    print('PREFLIGHT_PASSED',timings,flush=True)


if __name__=='__main__':
    main()
