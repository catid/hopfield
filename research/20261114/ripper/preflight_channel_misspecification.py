"""H1cd synthetic runner, missing-cost, seed, proof and endpoint gates."""
from pathlib import Path
import json,platform
import numpy as np
import channel_misspecification_study as S
from audit_channel_misspecification import aggregate,empty_grid
from channel_misspecification_math import score,independent_score,CHANNELS,UncertifiedCost
from degree_prior_solver import solve
from degree_prior_math import attempt_certificate
from audit_degree_prior import check_certificate


def seed_audit(design):
    values=set();files={}
    def scan(value,context=False):
        if isinstance(value,dict):
            for key,item in value.items():scan(item,context or 'seed' in key.lower())
        elif isinstance(value,list):
            for item in value:scan(item,context)
        elif context and type(value) is int:values.add(value)
    for path in sorted((S.ROOT/'research').rglob('*.json')):
        try:data=S.read(path)
        except (UnicodeDecodeError,json.JSONDecodeError):continue
        if isinstance(data,dict) and 'frozen_utc' in data:
            files[str(path.relative_to(S.ROOT))]=S.sha(path);scan(data)
    known={}
    for name,count in [('fixed-270-confirm',128),('repeat-query-tradeoff',128),('raw-count-quantization',128),('posterior-query-labels',128),('posterior-fresh-confirm',192),('pair-likelihood',128)]:
        parent=S.read(S.HERE/f'protocol-{name}.json');base=parent['solver']['random_seed_base']
        known[name]={base+41*i+d for i in range(count) for d in range(1,41)}
    base=design['candidate_solver_seed_base'];proposed={base+41*i+d for i in range(128) for d in range(1,41)}
    assert len(proposed)==5120 and not proposed&values and not proposed&set().union(*known.values()) and 0<min(proposed)<max(proposed)<2**31
    receipt=dict(status='passed',frozen_protocol_sha256=files,historical_explicit_seed_count=len(values),
                 expanded_solver_ranges={name:[min(v),max(v)] for name,v in known.items()},expanded_solver_counts={name:len(v) for name,v in known.items()},
                 new_solver_seed_range=[min(proposed),max(proposed)],new_solver_seed_count=5120,collisions=[],
                 scope='All explicit integer seed fields plus expanded H1bv/H1bw/H1by/H1bz/H1ca/H1cc actual solver seeds. Other historical formulas not exhaustive; no PRNG independence proof.')
    S.save(S.HERE/'channel-misspecification-seed-audit.json',receipt);return receipt


def fixture(kinds,selected_arm):
    assert len(kinds)==64;rows=[]
    for seed in range(64):
        for ai,arm in enumerate(S.ARMS):
            kind=kinds[seed] if ai==selected_arm else 'success'
            rows.append(dict(index=2*seed+ai,seed=seed,arm=arm,old_exact=seed not in [7,57],classification=kind,
                             selected_degree=20,fixed_degree20=dict(classification=kind),top810=dict(classification=kind),numeric=dict(fit_allowed=kind!='unresolved')))
    return aggregate(rows)


def main():
    assert platform.node()=='ripper'
    target=S.HERE/'channel-misspecification-preflight.json'
    assert not target.exists() and not S.PROTOCOL.exists() and not S.OUT.exists() and not S.PUBLIC.exists()
    design=S.read(S.HERE/'channel-misspecification-proposal.json');seeds=seed_audit(design)
    mathpath=S.HERE/'channel-misspecification-math-check.json';checked=S.read(mathpath)
    assert checked['status']=='passed' and not checked['study_scores_read'] and checked['host']=='ripper'
    for name,digest in checked['source_sha256'].items():assert S.sha(S.HERE/name)==digest
    import run_channel_misspecification as runner
    assert runner.OUT==S.OUT and runner.PUBLIC==S.PUBLIC and runner.PROTOCOL==S.PROTOCOL
    for action,seed in [('summary',None)]+[('worker',i) for i in range(8)]:
        cmd=runner.command(action,seed)
        assert cmd[1]==str(S.HERE/'channel_misspecification_study.py') and Path(cmd[1]).exists() and cmd[2]==action
        assert cmd[3:]==([] if seed is None else ['--seed',str(seed)])
    good,bad,frac,unknown='success','resolved_failure_nonexact','resolved_failure_fractional','unresolved'
    cases=[({},('supported','supported')),
        ({7:bad,57:bad,0:bad,1:frac},('supported','supported')),
        ({7:bad,57:bad,0:bad,1:bad,2:bad},('falsified','falsified')),
        ({0:bad,1:bad,2:bad},('supported','falsified')),
        ({0:unknown,1:unknown},('supported','supported')),
        ({0:unknown,1:unknown,2:unknown},('supported','inconclusive')),
        ({7:bad,57:bad,0:bad,1:bad,2:unknown},('inconclusive','inconclusive')),
        ({7:bad,0:bad,1:bad,2:bad,3:unknown},('inconclusive','falsified')),
        ({i:unknown for i in range(64)},('inconclusive','inconclusive'))]
    endpoints=[]
    for ai,arm in enumerate(S.ARMS):
        for changes,expected in cases:
            kinds=[changes.get(seed,good) for seed in range(64)];result=fixture(kinds,ai)
            assert tuple(result['endpoints'][arm][key]['decision'] for key in ['exact','regressions'])==expected
            assert all(row['decision']=='supported' for row in result['endpoints'][S.ARMS[1-ai]].values())
            endpoints.append(result)
    designarms={arm['name']:(arm['assumed_noise_numerator'],arm['assumed_noise_denominator']) for arm in design['arms']}
    assert CHANNELS==designarms and S.ARMS==list(CHANNELS)
    synthetic=dict(first=np.arange(270*81,dtype=np.int64).reshape(270,81)%9,given=np.ones((270,81),bool),
                   repeat1=np.ones((90,81),np.int8),repeat2=np.zeros((90,81),np.int8))
    left=S.arm_queries(synthetic,S.ARMS[0]);right=S.arm_queries(synthetic,S.ARMS[1])
    assert all(np.array_equal(a,b) for a,b in zip(left,right)) and left[0].shape==(3,90,81) and left[1].shape==(90,81)
    original=S.independent_score
    def uncertified(*args,**kwargs):raise UncertifiedCost('synthetic unproven rounding')
    S.independent_score=uncertified
    try:
        failure=S.independent_numeric(None,None,CHANNELS[S.ARMS[0]])
        assert failure==dict(certified=False,output=None,error='synthetic unproven rounding')
    finally:S.independent_score=original
    real_solver=S.solve
    def forbidden(*args,**kwargs):raise AssertionError('Uncertified zero placeholder reached solver')
    S.solve=forbidden
    try:
        for producer,reference in [(False,True),(True,False),(False,False)]:
            arrays=dict(cost=np.zeros(3240,np.int64),cost_certified=np.full(3240,True,bool),cost_precision=np.full(3240,80,np.int64))
            if not producer:arrays['cost_certified'][137]=False
            state=S.numeric_state(arrays,dict(certified=reference,error=None if reference else 'synthetic_reference_interval_failure'))
            assert not state['fit_allowed']
            actual,records=S.fit_models(arrays,state,0,{},Path('/unused-synthetic-only'),S.ARMS[0])
            assert actual is arrays and records==[] and 'top810_support' not in actual
            assert not any(key.startswith('d') and key[1:3].isdigit() for key in actual)
    finally:S.solve=real_solver
    unknown_grid=empty_grid();assert unknown_grid['selected_degree'] is None and unknown_grid['certified_grid_LPs']==0 and not unknown_grid['analytic_zero_proof'] and unknown_grid['unresolved_degrees']==list(range(1,41))
    folder=S.ROOT/'artifacts/research/20261114/ripper/channel_misspecification_preflight';folder.mkdir(parents=True,exist_ok=True)
    toy=np.array([[[0,0,1,1]],[[0,1,1,2]],[[1,0,2,1]]],np.int8);given=np.ones((1,4),bool)
    certificates=[]
    for ai,arm in enumerate(S.ARMS):
        scored=score(toy,given,CHANNELS[arm]);cost,joint,_=independent_score(toy,given,CHANNELS[arm])
        assert scored['cost_certified'].all() and np.array_equal(scored['cost'],cost) and np.array_equal(scored['joint_count'],joint)
        ii,jj=np.triu_indices(4,1);x,y,meta=solve(4,1,ii,jj,cost,992203+ai-2026111502,folder/f'{arm}.log')
        cert,witness=attempt_certificate(4,1,ii,jj,cost,x,y);proof=[witness[key] for key in ['primal_twice','node_dual_twice','upper_dual_twice']]
        assert meta['solver_optimal'] and cert['certified'] and check_certificate(4,1,ii,jj,cost,*proof)
        for which in range(3):
            altered=[v.copy() for v in proof];altered[which][0]+=1;assert not check_certificate(4,1,ii,jj,cost,*altered)
        certificates.append(cert)
    partitions=[list(range(worker,64,8)) for worker in range(8)];assert sorted(b for part in partitions for b in part)==list(range(64))
    sources={S.ROOT/rel for rel in S.read(S.HERE/'pair-likelihood-preflight.json')['source_sha256']}
    sources|={S.HERE/name for name in ['channel_misspecification_math.py','check_channel_misspecification_math.py','channel_misspecification_study.py','audit_channel_misspecification.py','preflight_channel_misspecification.py','run_channel_misspecification.py']}
    receipt=dict(status='passed',host=platform.node(),study_scores_read=False,proposal_sha256=S.sha(S.HERE/'channel-misspecification-proposal.json'),math_gate_sha256=S.sha(mathpath),seed_audit_sha256=S.sha(S.HERE/'channel-misspecification-seed-audit.json'),
                 runner_import_and_all9_command_bindings_verified=True,all_uncertified_cost_modes_skip_entire_grid=True,numeric_placeholder_solver_calls=0,
                 endpoint_cases=endpoints,synthetic_LP_certificates=certificates,mutations_rejected=6,
                 arm_channel_bindings=True,identical_queries_across_arms=True,independent_interval_exception_retained=True,worker_partition=partitions,
                 source_sha256={str(path.relative_to(S.ROOT)):S.sha(path) for path in sorted(sources)})
    S.save(target,receipt);print(json.dumps(dict(status='passed',source_files=len(sources),solver_range=seeds['new_solver_seed_range'])))


if __name__=='__main__':main()
