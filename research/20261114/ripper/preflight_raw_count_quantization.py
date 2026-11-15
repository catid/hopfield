"""Synthetic H1by affine objective/curvature, quantization, proof and endpoint gates."""
from fractions import Fraction
import json, platform, time
import numpy as np
from raw_count_quantization import ROOT, HERE, PROTOCOL, OUT, PUBLIC, ARMS, sha, save, raw_cost
from audit_raw_count_quantization import aggregate
from audit_degree_prior import check_certificate
from audit_laplace_rate import independent_rate_costs
from degree_prior_solver import solve
from degree_prior_math import attempt_certificate


def seed_audit(design):
    files, values = {}, set()
    def scan(value, context=False):
        if isinstance(value, dict):
            for key, item in value.items():
                scan(item, context or 'seed' in key.lower())
        elif isinstance(value, list):
            for item in value:
                scan(item, context)
        elif context and isinstance(value, int) and not isinstance(value, bool):
            values.add(value)
    for path in sorted((ROOT/'research').rglob('*.json')):
        try:
            data = json.loads(path.read_text())
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and 'frozen_utc' in data:
            files[str(path.relative_to(ROOT))] = sha(path)
            scan(data)
    proposed = {design['solver_seed_base']+41*i+d for i in range(128) for d in range(1, 41)}
    assert len(proposed) == 5120 and not proposed & values
    result = dict(status='passed', historical_explicit_seed_count=len(values), frozen_protocol_sha256=files,
                  solver_seed_range=[min(proposed), max(proposed)], new_solver_seeds=5120, collisions=[],
                  scope='Explicit integer seed-valued fields only; does not expand every historical formula or prove stream independence.')
    save(HERE/'raw-count-quantization-seed-audit.json', result)
    return result


def fixture(triple_kinds):
    assert len(triple_kinds) == 64
    rows = []
    for seed in range(64):
        for ai, arm in enumerate(ARMS):
            kind = 'success' if arm == 'single270' else triple_kinds[seed]
            rows.append(dict(index=2*seed+ai, seed=seed, arm=arm, old_exact=arm=='single270' or seed<50,
                             classification=kind, selected_degree=20, fixed_degree20=dict(classification=kind), top810=dict(classification=kind)))
    return aggregate(rows)


def main():
    assert platform.node() == 'ripper'
    target = HERE/'raw-count-quantization-preflight.json'
    assert not target.exists() and not PROTOCOL.exists() and not OUT.exists() and not PUBLIC.exists()
    design = json.loads((HERE/'raw-count-quantization-proposal.json').read_text())
    seeds = seed_audit(design)
    good, bad, frac, unknown = 'success', 'resolved_failure_nonexact', 'resolved_failure_fractional', 'unresolved'
    cases = [
        ([good]*53+[bad]*11, ('supported','supported')),
        ([good]*52+[bad]*12, ('falsified','supported')),
        ([good]*52+[unknown]+[bad]*11, ('inconclusive','supported')),
        ([bad]+[good]*53+[bad]*10, ('supported','supported')),
        ([bad,frac]+[good]*53+[bad]*9, ('supported','falsified')),
        ([bad,unknown]+[good]*52+[bad]*10, ('inconclusive','inconclusive')),
    ]
    endpoints = []
    for kinds, decisions in cases:
        result = fixture(kinds)
        assert tuple(result['endpoints'][key]['decision'] for key in ['net_triple_gain','triple_regression']) == decisions
        endpoints.append(result)
    assert endpoints[3]['paired_changes']['triple90']['losses'] == [0]
    assert endpoints[3]['paired_changes']['triple90']['gains'] == [50,51,52,53]
    # Derive affine identity with exact Fractions on feasible integral/fractional examples.
    rng = np.random.default_rng(988001)
    affine = []
    for n, degree, twice in [(4,1,np.array([2,0,0,0,0,2],np.int64)),
                              (3,1,np.ones(3,np.int64)),
                              (4,2,np.array([2,2,0,0,2,2],np.int64))]:
        ii,jj=np.triu_indices(n,1)
        node=np.zeros(n,np.int64);np.add.at(node,ii,twice);np.add.at(node,jj,twice)
        assert (node==2*degree).all()
        for m in [71,213]:
            counts=rng.integers(0,m+1,len(twice),dtype=np.int64)
            direct=sum(Fraction(int(c)+1,m+2)*Fraction(int(x),2) for c,x in zip(counts,twice))
            raw=sum(Fraction(int(c)*int(x),2) for c,x in zip(counts,twice))
            assert direct==(raw+Fraction(n*degree,2))/(m+2)
            affine.append(dict(n=n,degree=degree,exposure=m,raw_objective=str(raw),unrounded_rate_objective=str(direct)))
    curve=[]
    for m in [71,213]:
        raw=[d*d for d in range(41)] # every second difference ties
        rates=[Fraction(2*value+81*d,2*(m+2)) for d,value in enumerate(raw)]
        raw_second=[raw[d+1]+raw[d-1]-2*raw[d] for d in range(1,40)]
        rate_second=[rates[d+1]+rates[d-1]-2*rates[d] for d in range(1,40)]
        assert all(b==Fraction(a,m+2) for a,b in zip(raw_second,rate_second))
        raw_best=min(range(1,40),key=lambda d:(-raw_second[d-1],d))
        rate_best=min(range(1,40),key=lambda d:(-rate_second[d-1],d))
        assert raw_best==rate_best==1
        curve.append(dict(exposure=m,lower_degree_tie=raw_best))
    # Two degree-one matchings tie in raw counts but not after rounding at m=71.
    counts=np.array([0,71,1,1,71,2],np.int64)
    truth=np.array([2,0,0,0,0,2],np.int64)
    alternative=np.array([0,0,2,2,0,0],np.int64)
    node=np.array([0,0,2,2],np.int64);upper=np.zeros(6,np.int64);ii,jj=np.triu_indices(4,1)
    for primal in [truth,alternative]:assert check_certificate(4,1,ii,jj,counts,primal,node,upper)
    rounded=independent_rate_costs(np.full(6,71,np.int64),counts)
    raw_gap=int(counts@truth-counts@alternative);rounded_gap=int(rounded@truth-rounded@alternative)
    assert raw_gap==0 and rounded_gap==2
    rounded_fixture=dict(raw_gap_twice=raw_gap,rounded_gap_twice=rounded_gap,raw_costs=counts.tolist(),rounded_costs=rounded.tolist())
    valid=np.arange(3240,dtype=np.int64)%72;joint=np.full(3240,71,np.int64)
    result=raw_cost(joint,valid,71);assert np.array_equal(result,valid) and not np.shares_memory(result,valid)
    bad_cases=[(joint.astype(np.int32),valid,71),(joint,valid.astype(np.int32),71),(joint[:-1],valid[:-1],71),(joint,valid,213),(joint,np.full(3240,-1,np.int64),71),(joint,np.full(3240,72,np.int64),71)]
    for args in bad_cases:
        try:raw_cost(*args)
        except AssertionError:pass
        else:raise AssertionError('Invalid count input accepted')
    folder=ROOT/'artifacts/research/20261114/ripper/raw_count_quantization_preflight';folder.mkdir(parents=True,exist_ok=True)
    x,y,metadata=solve(4,1,ii,jj,counts,988002-2026111502,folder/'toy.log')
    cert,w=attempt_certificate(4,1,ii,jj,counts,x,y);proof=[w[key] for key in ['primal_twice','node_dual_twice','upper_dual_twice']]
    assert metadata['solver_optimal'] and cert['certified'] and cert['primal_objective_twice']==4 and check_certificate(4,1,ii,jj,counts,*proof)
    for k in range(3):
        changed=[v.copy() for v in proof];changed[k][0]+=1
        assert not check_certificate(4,1,ii,jj,counts,*changed)
    started=time.perf_counter()
    for _ in range(1000):assert np.array_equal(raw_cost(joint,valid,71),valid)
    elapsed=time.perf_counter()-started
    inherited=json.loads((HERE/'repeat-query-tradeoff-preflight.json').read_text())['source_sha256']
    files={ROOT/rel for rel in inherited}|{HERE/name for name in ['raw_count_quantization.py','audit_raw_count_quantization.py','run_raw_count_quantization.py','preflight_raw_count_quantization.py']}
    output=dict(status='passed',study_rawcount_scores_read=False,affine_objective_fixtures=affine,curvature_affine_and_tie_fixtures=curve,
                rounded_tie_counterexample=rounded_fixture,endpoint_cases=endpoints,invalid_count_cases_rejected=len(bad_cases),synthetic_LP_certificate=cert,
                count_validation_1000_calls_wall_seconds=elapsed,performance_scope='Synthetic3240pair count validation/copy only; no full scientific speedup claim.',
                seed_audit_sha256=sha(HERE/'raw-count-quantization-seed-audit.json'),source_sha256={str(path.relative_to(ROOT)):sha(path) for path in sorted(files)})
    save(target,output)
    print(json.dumps(dict(status='passed',source_files=len(files),solver_seed_range=seeds['solver_seed_range'],synthetic_count_validation_seconds=elapsed)))


if __name__=='__main__':
    main()
