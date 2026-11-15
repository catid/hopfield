"""Synthetic H1ca sources, masks, records, three-arm inference and endpoint gates."""
import json,platform
from itertools import product
import numpy as np
from posterior_fresh_confirm_study import ROOT,HERE,PROTOCOL,OUT,PUBLIC,REGISTRY,PARENT,ARMS,ARM_ROWS,sha,save,npz,make_masks,reconstruct,arm_values,arm_score
from audit_posterior_fresh_confirm import aggregate,literal_account,reference_masks,reference_observations,score_reference
from posterior_query_math import score,independent_score
from degree_prior_solver import solve
from degree_prior_math import attempt_certificate
from audit_degree_prior import check_certificate


def seed_audit(design):
    files,values={},set()
    def scan(value,context=False):
        if isinstance(value,dict):
            for key,item in value.items():scan(item,context or 'seed' in key.lower())
        elif isinstance(value,list):
            for item in value:scan(item,context)
        elif context and isinstance(value,int) and not isinstance(value,bool):values.add(value)
    for path in sorted((ROOT/'research').rglob('*.json')):
        try:data=json.loads(path.read_text())
        except (UnicodeDecodeError,json.JSONDecodeError):continue
        if isinstance(data,dict) and 'frozen_utc' in data:
            files[str(path.relative_to(ROOT))]=sha(path);scan(data)
    expanded={}
    for name in ['fixed-270-confirm','repeat-query-tradeoff','raw-count-quantization','posterior-query-labels']:
        p=json.loads((HERE/f'protocol-{name}.json').read_text())
        if name in ['fixed-270-confirm','repeat-query-tradeoff']:
            for stream,base in p['seeds'].items():
                if stream!='solver_base':expanded[name+'/'+stream]=set(range(base,base+64))
        base=p['solver']['random_seed_base']
        expanded[name+'/solver']={base+41*i+d for i in range(128) for d in range(1,41)}
    proposed={name:set(range(design['seeds'][name],design['seeds'][name]+64)) for name in ['source','balanced_mask','noise','repeat1','repeat2']}
    proposed['solver']={design['seeds']['solver_base']+41*i+d for i in range(192) for d in range(1,41)}
    combined=set().union(*proposed.values());known=set().union(*expanded.values())
    assert sum(len(v) for v in proposed.values())==len(combined)==8000
    assert not combined&values and not combined&known
    result=dict(status='passed',frozen_protocol_sha256=files,historical_explicit_seed_count=len(values),
                new_ranges={name:[min(v),max(v)] for name,v in proposed.items()},new_counts={name:len(v) for name,v in proposed.items()},
                historical_expanded_ranges={name:[min(v),max(v)] for name,v in expanded.items()},
                historical_expanded_counts={name:len(v) for name,v in expanded.items()},collisions=[],
                scope='All explicit historical integer seed fields plus actual64-bank H1bv/H1bw source/mask/noise/repeat streams and all5120 actual solver seeds each for H1bv/H1bw/H1by/H1bz. Other formulas are not exhaustive; no PRNG independence proof.')
    save(HERE/'posterior-fresh-confirm-seed-audit.json',result)
    return result


def fixture(triples):
    assert len(triples)==64
    rows=[]
    for seed,kinds in enumerate(triples):
        for ai,arm in enumerate(ARMS):
            kind=kinds[ai]
            rows.append(dict(index=3*seed+ai,seed=seed,arm=arm,rows=ARM_ROWS[arm],classification=kind,selected_degree=20,
                             fixed_degree20=dict(classification=kind),top810=dict(classification=kind)))
    return aggregate(rows)


def endpoint_gates():
    good,bad,frac,unknown='success','resolved_failure_nonexact','resolved_failure_fractional','unresolved'
    states=[good,bad,frac,unknown]
    for left,hard,right in product(states,repeat=3):
        answer=fixture([(good,good,good)]*63+[(left,hard,right)])
        options=lambda kind:[0,1] if kind==unknown else [int(kind==good)]
        for key,control in [('versus_single270',left),('versus_hard_triple90',hard)]:
            actual=[a*(1-b) for a,b in product(options(control),options(right))]
            assert (answer['paired_changes'][key]['loss_lower'],answer['paired_changes'][key]['loss_upper'])==(min(actual),max(actual))
    cases=[([(good,bad,good)]*61+[(good,good,bad)]*3,('supported','supported')),
           ([(good,good,good)]*60+[(good,bad,frac)]*4,('falsified','falsified')),
           ([(good,bad,good)]*60+[(good,good,bad)]*3+[(good,good,unknown)],('inconclusive','inconclusive')),
           ([(good,good,good)]*60+[(good,good,bad)]*3+[(unknown,bad,bad)],('falsified','inconclusive')),
           ([(good,good,good)]*60+[(bad,bad,good)]*4,('supported','supported'))]
    receipts=[]
    for triples,expected in cases:
        result=fixture(triples)
        assert tuple(result['endpoints'][key]['decision'] for key in ['posterior_exact','paired_loss'])==expected
        receipts.append(result)
    return receipts


def main():
    assert platform.node()=='ripper'
    target=HERE/'posterior-fresh-confirm-preflight.json'
    assert not target.exists() and not PROTOCOL.exists() and not OUT.exists() and not REGISTRY.exists() and not PUBLIC.exists()
    design=json.loads((HERE/'posterior-fresh-confirm-proposal.json').read_text());seeds=seed_audit(design)
    parent=json.loads((HERE/'protocol-posterior-query-labels.json').read_text())
    math_hash=sha(HERE/'posterior_query_math.py')
    assert math_hash==parent['source_sha256']['research/20261114/ripper/posterior_query_math.py']=='e9dc113e5403b07884dd7c35dceb9d1efacd4f3d0a5781db65ea975e8da1f059'
    math_gate=json.loads((HERE/'posterior-query-math-check.json').read_text())
    assert math_gate['status']=='passed' and math_gate['host']=='ripper'
    for name,digest in math_gate['source_sha256'].items():assert sha(HERE/name)==digest
    prior=npz(PARENT/'excluded_ids.npz');selected=npz(PARENT/'selected_sources.npz')['all_source_row']
    excluded=np.union1d(prior['excluded'],selected);ripper=np.union1d(prior['ripper_prior'],selected)
    assert len(excluded)==design['prior_combined_excluded_ids']==2414270
    assert len(ripper)==design['prior_ripper_ledger_ids']==2177845 and np.isin(ripper,excluded).all()
    assert design['pool_size']-len(excluded)==design['eligible_rows']==1417724
    rng=np.random.default_rng(990101);clean=rng.integers(0,9,(270,81),dtype=np.int8)
    mask=make_masks(990102);assert np.array_equal(mask,reference_masks(990102))
    for n in [90,270]:
        matrix=mask[:n].astype(np.int64).T@mask[:n].astype(np.int64);k=n//90
        assert np.array_equal(matrix,71*k*np.ones((81,81),np.int64)+9*k*np.eye(81,dtype=np.int64))
    noise={}
    for name,n in [('first',270),('repeat1',90),('repeat2',90)]:
        noise[name+'_iid_uniforms']=rng.random((n,81));noise[name+'_iid_offsets']=rng.integers(1,9,(n,81),dtype=np.int8)
    records=reconstruct(mask,clean,noise);original,queries,decoded=reference_observations(mask,clean,noise)
    for name,n in [('first',270),('repeat1',90),('repeat2',90)]:
        assert np.array_equal(records[name][mask[:n]],queries[name][mask[:n]]) and (records[name][~mask[:n]]==-1).all()
    changed=((clean+(np.arange(81)%8+1)[None,:])%9).astype(np.int8);altered=reconstruct(mask,changed,noise)
    arm_checks={}
    for arm in ARMS:
        arrays=arm_score(records,arm);cost,joint,extra=score_reference(records,arm)
        assert np.array_equal(arrays['cost'],cost) and np.array_equal(arrays['joint_count'],joint)
        n=ARM_ROWS[arm];assert (joint==71*(n//90)).all()
        values,given=arm_values(records,arm);replacement,given2=arm_values(altered,arm)
        assert np.array_equal(given,given2)
        if arm=='posterior_triple90':assert (values[:,given]!=replacement[:,given]).all()
        else:
            assert (values[given]!=replacement[given]).all()
            old='single270' if arm=='single270' else 'triple90'
            assert np.array_equal(values,original[old]['values']) and np.array_equal(extra,original[old]['equal_count'])
        modified=arm_score(altered,arm)
        assert np.array_equal(joint,modified['joint_count']) and not np.array_equal(cost,modified['cost'])
        arm_checks[arm]=dict(pair_coqueries=int(joint[0]),queried_records=int(given.sum())*(1 if arm=='single270' else 3),
                             exact_independent_score_parity=True,all_queried_labels_substituted=True,costs_changed=True)
    accounting=literal_account(mask)
    assert accounting['full_physical_noisy_union']==32400 and accounting['shared_two_triple_methods']==19440
    assert accounting['single270_deployment']==accounting['hard_triple90_deployment']==accounting['posterior_triple90_deployment']==19440
    # Literal small posterior LP; original-method parity is independently checked above.
    folder=ROOT/'artifacts/research/20261114/ripper/posterior_fresh_confirm_preflight';folder.mkdir(parents=True,exist_ok=True)
    toy=np.array([[[0,0,1,1]],[[0,1,1,2]],[[1,0,2,1]]],np.int8);given=np.ones((1,4),bool)
    cost=score(toy,given)['cost'];literal,_,_=independent_score(toy,given,True);assert np.array_equal(cost,literal)
    ii,jj=np.triu_indices(4,1);x,y,metadata=solve(4,1,ii,jj,cost,990103-2026111502,folder/'posterior.log')
    cert,witness=attempt_certificate(4,1,ii,jj,cost,x,y);proof=[witness[key] for key in ['primal_twice','node_dual_twice','upper_dual_twice']]
    assert metadata['solver_optimal'] and cert['certified'] and check_certificate(4,1,ii,jj,cost,*proof)
    for k in range(3):
        bad=[v.copy() for v in proof];bad[k][0]+=1;assert not check_certificate(4,1,ii,jj,cost,*bad)
    partitions=[list(range(worker,64,8)) for worker in range(8)]
    assert sorted(bank for part in partitions for bank in part)==list(range(64))
    indices=[3*bank+ai for bank in range(64) for ai in range(3)];assert indices==list(range(192))
    files={ROOT/rel for rel in json.loads((HERE/'posterior-query-labels-preflight.json').read_text())['source_sha256']}
    files|={HERE/name for name in ['posterior_fresh_confirm_study.py','audit_posterior_fresh_confirm.py','preflight_posterior_fresh_confirm.py','run_posterior_fresh_confirm.py']}
    receipt=dict(status='passed',host=platform.node(),fresh_study_draws_performed=False,study_LPs_performed=False,
                 math_gate_sha256=sha(HERE/'posterior-query-math-check.json'),posterior_math_sha256=math_hash,
                 seed_audit_sha256=sha(HERE/'posterior-fresh-confirm-seed-audit.json'),exact_channel_sha256=sha(HERE/'repeat-query-channel.json'),
                 source_ledger=dict(combined=len(excluded),ripper=len(ripper),eligible=design['eligible_rows']),
                 full_270_90_mask_noise_records_parity=True,arm_checks=arm_checks,query_accounting_per_bank=accounting,
                 complete_label_substitution=True,endpoint_cases=endpoint_gates(),endpoint_state_combinations=64,
                 synthetic_LP_certificate=cert,mutations_rejected=3,worker_partition=partitions,grid_indices=indices,
                 source_sha256={str(path.relative_to(ROOT)):sha(path) for path in sorted(files)})
    save(target,receipt)
    print(json.dumps(dict(status='passed',source_files=len(files),new_seed_ranges=seeds['new_ranges'],posterior_math_sha256=math_hash)))


if __name__=='__main__':main()
