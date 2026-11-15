"""Synthetic H1bz query preparation, posterior rates, proofs and endpoint gates."""
import json, platform
import numpy as np
from posterior_query_labels import ROOT,HERE,PROTOCOL,OUT,PUBLIC,ARMS,sha,save,arm_queries
from audit_posterior_query_labels import aggregate
from prepare_posterior_queries import reconstruct
from repeat_query_tradeoff_study import make_masks
from audit_higher_noise import reference_corrupt
from audit_repeat_query_tradeoff import reference_decode
from posterior_query_math import score,independent_score
from degree_prior_solver import solve
from degree_prior_math import attempt_certificate
from audit_degree_prior import check_certificate


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
    previous = json.loads((HERE/'protocol-raw-count-quantization.json').read_text())
    known_actual = {previous['solver_seed_base']+41*i+d for i in range(128) for d in range(1,41)}
    assert len(proposed) == 5120 and not proposed & values and not proposed & known_actual
    result = dict(status='passed', historical_explicit_seed_count=len(values), frozen_protocol_sha256=files,
                  solver_seed_range=[min(proposed), max(proposed)], new_solver_seeds=5120, collisions=[], expanded_H1by_solver_seed_range=[min(known_actual),max(known_actual)],
                  scope='Explicit integer seed-valued fields plus all5120 actualH1by solver seeds; other formulas not exhaustively expanded, no PRNG independence proof.')
    save(HERE/'posterior-query-labels-seed-audit.json', result)
    return result


def fixture(kinds):
    assert len(kinds)==64
    rows=[]
    for bank in range(64):
        for ai,arm in enumerate(ARMS):
            kind='success' if ai==0 else kinds[bank]
            rows.append(dict(index=2*bank+ai,seed=bank,arm=arm,old_exact=ai==0 or bank<50,classification=kind,
                             selected_degree=20,fixed_degree20=dict(classification=kind),top810=dict(classification=kind)))
    return aggregate(rows)


def main():
    assert platform.node()=='ripper'
    target=HERE/'posterior-query-labels-preflight.json'
    assert not target.exists() and not PROTOCOL.exists() and not OUT.exists() and not PUBLIC.exists()
    design=json.loads((HERE/'posterior-query-labels-proposal.json').read_text())
    seeds=seed_audit(design)
    checked=json.loads((HERE/'posterior-query-math-check.json').read_text())
    assert checked['status']=='passed' and checked['host']=='ripper'
    for name,digest in checked['source_sha256'].items():assert sha(HERE/name)==digest
    good,bad,unknown,frac='success','resolved_failure_nonexact','unresolved','resolved_failure_fractional'
    cases=[([good]*58+[bad]*6,('supported','supported')),
           ([good]*57+[bad]*7,('falsified','supported')),
           ([good]*57+[unknown]+[bad]*6,('inconclusive','supported')),
           ([bad]+[good]*58+[bad]*5,('supported','supported')),
           ([bad,frac]+[good]*58+[bad]*4,('supported','falsified')),
           ([bad,unknown]+[good]*57+[bad]*5,('inconclusive','inconclusive'))]
    endpoints=[]
    for kinds,expect in cases:
        result=fixture(kinds)
        assert tuple(result['endpoints'][key]['decision'] for key in ['triple_exact','triple_regression'])==expect
        endpoints.append(result)
    rng=np.random.default_rng(989101)
    clean=rng.integers(0,9,(270,81),dtype=np.int8)
    mask=make_masks(989102)
    noise={}
    for name,n in [('first',270),('repeat1',90),('repeat2',90)]:
        noise[name+'_iid_uniforms']=rng.random((n,81))
        noise[name+'_iid_offsets']=rng.integers(1,9,(n,81),dtype=np.int8)
    records=reconstruct(mask,clean,noise)
    literal={}
    for name,n in [('first',270),('repeat1',90),('repeat2',90)]:
        full=reference_corrupt(clean[:n],{key:noise[name+'_'+key] for key in ['iid_uniforms','iid_offsets']})
        literal[name]=full
        assert np.array_equal(records[name][mask[:n]],full[mask[:n]]) and (records[name][~mask[:n]]==-1).all()
    decoded=reference_decode(literal['first'][:90],literal['repeat1'],literal['repeat2'])
    from repeat_query_math import observations
    assert np.array_equal(observations(mask,clean,noise)['triple90']['values'][mask[:90]],decoded[mask[:90]])
    results={}
    for arm in ARMS:
        values,given=arm_queries(records,arm)
        produced=score(values,given)
        independent,joint,_=independent_score(values,given)
        assert np.array_equal(produced['cost'],independent) and np.array_equal(produced['joint_count'],joint)
        assert (joint==(213 if arm=='single270' else 71)).all()
        results[arm]=dict(queried_records=int(values.shape[0]*given.sum()),pair_coqueries=int(joint[0]))
    assert results['single270']['queried_records']==results['triple90']['queried_records']==19440
    assert 19440+19440-6480==32400
    folder=ROOT/'artifacts/research/20261114/ripper/posterior_query_labels_preflight';folder.mkdir(parents=True,exist_ok=True)
    toy=np.array([[[0,0,1,1]],[[0,1,1,2]],[[1,0,2,1]]],np.int8);given=np.ones((1,4),bool)
    cost=score(toy,given)['cost'];ii,jj=np.triu_indices(4,1)
    x,y,metadata=solve(4,1,ii,jj,cost,989103-2026111502,folder/'posterior.log')
    cert,w=attempt_certificate(4,1,ii,jj,cost,x,y);proof=[w[key] for key in ['primal_twice','node_dual_twice','upper_dual_twice']]
    assert metadata['solver_optimal'] and cert['certified'] and check_certificate(4,1,ii,jj,cost,*proof)
    for k in range(3):
        badproof=[v.copy() for v in proof];badproof[k][0]+=1
        assert not check_certificate(4,1,ii,jj,cost,*badproof)
    files={ROOT/rel for rel in json.loads((HERE/'repeat-query-tradeoff-preflight.json').read_text())['source_sha256']}
    files|={HERE/name for name in ['posterior_query_math.py','check_posterior_query_math.py','posterior_query_labels.py','audit_posterior_query_labels.py',
                                  'prepare_posterior_queries.py','preflight_posterior_query_labels.py','run_posterior_query_labels.py']}
    receipt=dict(status='passed',study_posterior_scores_read=False,synthetic_preparation_correspondence=True,
                 full_270_90_exact_posterior_parity=True,arm_query_accounting=results,physical_query_union_per_bank=32400,
                 endpoint_cases=endpoints,synthetic_LP_certificate=cert,mutations_rejected=3,
                 math_gate_sha256=sha(HERE/'posterior-query-math-check.json'),seed_audit_sha256=sha(HERE/'posterior-query-labels-seed-audit.json'),
                 source_sha256={str(path.relative_to(ROOT)):sha(path) for path in sorted(files)})
    save(target,receipt)
    print(json.dumps(dict(status='passed',source_files=len(files),solver_seed_range=seeds['solver_seed_range'])))


if __name__=='__main__':
    main()


