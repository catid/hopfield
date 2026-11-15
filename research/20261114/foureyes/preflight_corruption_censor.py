"""H4at synthetic count, invariance, nonuniqueness, LP and endpoint gates."""
from pathlib import Path
import json
import time
import numpy as np
import corruption_censor as S
from corruption_censor_math import score, dense_score, blocked_score, independent_score, literal_score, mask_from_indices
from audit_corruption_censor import aggregate, selection_stream
from degree_prior_solver import solve
from degree_prior_math import attempt_certificate
from audit_degree_prior import check_certificate


def seeds():
    values, files = set(), {}
    def scan(value, context=False):
        if isinstance(value, dict):
            for key, item in value.items():
                scan(item, context or 'seed' in key.lower())
        elif isinstance(value, list):
            for item in value:
                scan(item, context)
        elif context and isinstance(value, int) and not isinstance(value, bool):
            values.add(value)
    for path in sorted((S.ROOT/'research').rglob('*.json')):
        try:
            data = json.loads(path.read_text())
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and 'frozen_utc' in data:
            files[str(path.relative_to(S.ROOT))] = S.sha(path)
            scan(data)
    new = set(range(S.SOLVER_SEED_BASE, S.SOLVER_SEED_BASE+256))
    assert not new & values
    result = dict(status='passed', explicit_seed_values=len(values), frozen_protocol_sha256=files,
                  new_solver_seeds=256, solver_seed_range=[min(new), max(new)], collisions=[],
                  scope='All explicit integer seed-valued fields in snapshotted frozen protocols. Does not expand every formula or establish undocumented-stream independence.')
    S.save(S.HERE/'corruption-censor-seed-audit.json', result)


def main():
    assert not S.PROTOCOL.exists() and not S.OUT.exists()
    started=time.perf_counter(); seeds(); rng=np.random.default_rng(982040)
    parity_cases=0
    for n, rows in [(4, 17), (11, 81), (81, 600)]:
        noisy=rng.integers(0,9,(rows,n),dtype=np.int8)
        for kind in ['none', 'all', 'mixed']:
            mask=np.zeros_like(noisy, bool) if kind=='none' else np.ones_like(noisy,bool) if kind=='all' else rng.random(noisy.shape)<.2
            expected=literal_score(noisy,mask)
            for fn in [score,blocked_score,independent_score]:
                result=fn(noisy,mask)
                assert all(np.array_equal(result[k],expected[k]) for k in expected)
            assert ((expected['cost']>=0)&(expected['cost']<=expected['exposure'])).all()
            if kind=='all':assert not expected['cost'].any() and not expected['exposure'].any()
            parity_cases+=1
    base=score(noisy,mask);ii,jj=np.triu_indices(81,1)
    vertex=rng.permutation(81);digit=rng.permutation(9)
    permuted=score(noisy[:,vertex],mask[:,vertex]);digits=score(digit[noisy],mask)
    for key in base:
        matrix=np.zeros((81,81),np.int64);matrix[ii,jj]=matrix[jj,ii]=base[key]
        assert np.array_equal(permuted[key],matrix[vertex][:,vertex][ii,jj])
        assert np.array_equal(digits[key],base[key])
    # Original producer versus independent full-stream mask selection on unrelated seeds.
    hot=rng.choice(81,18,replace=False)
    hot=np.random.default_rng(982041).choice(81,18,replace=False)
    uniforms=1-np.random.default_rng(982042).random((2400,81));weights=np.ones(81);weights[hot]=32
    indices=np.argsort(-np.log(uniforms)/weights,axis=1,kind='stable')[:600,:9]
    assert np.array_equal(indices,selection_stream(982041,982042))
    mask=mask_from_indices(indices);assert (mask.sum(1)==9).all()
    for broken in [indices.astype(np.int32),np.zeros_like(indices),np.full_like(indices,81)]:
        try:mask_from_indices(broken)
        except AssertionError:pass
        else:raise AssertionError('Malformed indices accepted')
    # True-edge zero counts do not imply a unique graph: two degree-one matchings.
    toy=np.array([[0,1,0,1]],np.int8);clean_mask=np.zeros_like(toy,bool)
    toy_cost=score(toy,clean_mask)['cost'];assert toy_cost.tolist()==[0,1,0,0,1,0]
    ti,tj=np.triu_indices(4,1);zero_node=np.zeros(4,np.int64);zero_upper=np.zeros(6,np.int64)
    truth=np.array([2,0,0,0,0,2],np.int64);alternative=np.array([0,0,2,2,0,0],np.int64)
    assert not np.array_equal(truth,alternative)
    for primal in [truth,alternative]:assert check_certificate(4,1,ti,tj,toy_cost,primal,zero_node,zero_upper)
    false_zeros=int(np.count_nonzero((toy_cost==0)&(truth==0)));assert false_zeros==2
    # Marked values may change arbitrarily without contaminating retained costs.
    marked=np.array([[True,False,False,True]])
    corrupt=toy.copy();corrupt[marked]=(corrupt[marked]+2)%9
    assert np.array_equal(corrupt[~marked],toy[~marked])
    assert not score(corrupt,marked)['cost'][truth==2].any()
    folder=S.ROOT/'artifacts/research/20261114/foureyes/corruption_censor_preflight';folder.mkdir(parents=True,exist_ok=True)
    certificates=[]
    for label,cost,expected in [('zero',toy_cost,0),('positive',np.ones(6,np.int64),4)]:
        x,y,metadata=solve(4,1,ti,tj,cost,982043-2026111502,folder/(label+'.log'))
        cert,proof=attempt_certificate(4,1,ti,tj,cost,x,y)
        assert metadata['solver_optimal'] and cert['certified'] and cert['primal_objective_twice']==expected
        w=[proof[k] for k in ['primal_twice','node_dual_twice','upper_dual_twice']]
        assert check_certificate(4,1,ti,tj,cost,*w)
        for k in range(3):
            changed=[v.copy() for v in w];changed[k][0]+=1
            assert not check_certificate(4,1,ti,tj,cost,*changed)
        certificates.append(cert)
    timing={};expected=literal_score(noisy,mask)
    for fn in [dense_score,blocked_score,independent_score,literal_score]:
        elapsed=[]
        for _ in range(3):
            t=time.perf_counter();answer=fn(noisy,mask);elapsed.append(time.perf_counter()-t)
            assert all(np.array_equal(answer[k],expected[k]) for k in answer)
        timing[fn.__name__+'_median_seconds']=float(np.median(elapsed))
    # Full synthetic denominators exercise independent loss and recovery rules.
    endpoints = []
    for successes, losses, expected in [(245, 8, ('supported', 'supported')), (244, 9, ('falsified', 'falsified'))]:
        rows = []
        retained = 129-losses
        gained = successes-retained
        for i in range(256):
            baseline = i < 129
            good = i < retained or 129 <= i < 129+gained
            rows.append(dict(index=i, family=S.FAMILIES[i % 4], baseline_exact=baseline,
                             classification='success' if good else 'resolved_failure_nonexact',
                             certificates_checked=2, mutations_rejected=6))
        result = aggregate(rows)
        assert (result['primary']['recovery']['decision'], result['primary']['paired_loss']['decision']) == expected
        endpoints.append(result['primary'])
    # Last fixture has 244 successes / 9 losses; one uncertain baseline loss
    # straddles both separate thresholds without changing the denominator.
    rows[120]['classification'] = 'unresolved'
    uncertain = aggregate(rows)['primary']
    assert uncertain['recovery'] == dict(lower=244, upper=245, required=245, decision='inconclusive')
    assert uncertain['paired_loss'] == dict(lower=8, upper=9, maximum=8, decision='inconclusive')
    endpoints.append(uncertain)
    files=[Path(__file__),S.HERE/'corruption_censor.py',S.HERE/'corruption_censor_math.py',S.HERE/'audit_corruption_censor.py',S.HERE/'audit_objective_gap.py']
    files += [S.HERE.parent/'ripper'/name for name in ['degree_prior_solver.py','degree_prior_math.py','audit_degree_prior.py']]
    S.save(S.HERE/'corruption-censor-preflight.json',dict(status='passed',study_censored_scores_read=False,synthetic_parity_cases=parity_cases,
        vertex_and_digit_permutation_parity=True,zero_exposure_pairs_retained=True,unchanged_observations_identity=True,
        full2400_stream_then600_prefix_verified=True,nonunique_zero_cost_fixture=dict(false_zero_edges=false_zeros,distinct_optimal_degree1_graphs=2),
        synthetic_certificates=certificates,endpoint_gates=endpoints,score_timings=timing,
        implementation_choice='Dense pair arrays; blocked implementation has no demonstrated worthwhile advantage on this bounded fixture. Independent audit uses direct pair row selection.',
        gate_wall_seconds=time.perf_counter()-started,seed_audit_sha256=S.sha(S.HERE/'corruption-censor-seed-audit.json'),
        source_sha256={str(p.relative_to(S.ROOT)):S.sha(p) for p in files}))
    print('PREFLIGHT_PASSED',json.dumps(timing),flush=True)


if __name__=='__main__':main()
