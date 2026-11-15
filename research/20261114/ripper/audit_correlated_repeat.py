"""Independent H1ch fresh observations, fixed-method costs, proofs and endpoints."""
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
from fractions import Fraction
import json,time
import numpy as np
from correlated_repeat_study import ROOT,HERE,OUT,PUBLIC,PROTOCOL,ARMS,RATE,RHO,PARAMETERS,read,sha,npz,verify,arm_values,numeric_state
from correlated_repeat_data import identity_audit,reference_masks,literal_account,channel_event
from correlated_repeat_math import score as supplied_score,independent_score as supplied_reference
from estimated_channel_math import score as estimated_score,independent_score as estimated_reference,estimate,independent_estimate
from pair_likelihood_math import rounded_decimal_interval,UncertifiedCost
from audit_partial_degree import grid
from audit_extra_coverage import statistics


def aggregate(rows):
    assert [r['index'] for r in rows]==list(range(144)) and [r['arm'] for r in rows]==ARMS*72
    groups=[];endpoints={}
    for ai,arm in enumerate(ARMS):
        part=rows[ai::2];assert [r['seed'] for r in part]==list(range(72))
        group=dict(arm=arm,**{kind:sum(r['classification']==kind for r in part) for kind in ['success','resolved_failure_nonexact','resolved_failure_fractional','unresolved']},
            selected_degree20=sum(r['selected_degree']==20 for r in part),fixed20_exact=sum(r['fixed_degree20']['classification']=='success' for r in part),top810_exact=sum(r['top810']['classification']=='success' for r in part),
            estimator_unresolved=sum(r['estimator'] is not None and not r['estimator']['fit_allowed'] for r in part),numerical_unresolved=sum(r['numeric'] is not None and not r['numeric']['fit_allowed'] for r in part))
        groups.append(group);lo=group['success'];hi=lo+group['unresolved']
        endpoints['supplied_exact' if ai==0 else 'iid_exact']=dict(lower=lo,upper=hi,required=60,total=72,decision='supported' if lo>=60 else 'falsified' if hi<60 else 'inconclusive')
    pairs={k:[] for k in ['gains','losses','both_exact','both_resolved_failure','any_unresolved']};low=high=0
    failures=['resolved_failure_nonexact','resolved_failure_fractional']
    for seed in range(72):
        left,right=[rows[2*seed+ai]['classification'] for ai in [0,1]]
        assert left in failures+['success','unresolved'] and right in failures+['success','unresolved']
        low+=left=='success' and right in failures;high+=left in ['success','unresolved'] and right!='success'
        key='any_unresolved' if 'unresolved' in [left,right] else 'both_exact' if left==right=='success' else 'gains' if right=='success' else 'losses' if left=='success' else 'both_resolved_failure'
        pairs[key].append(seed)
    endpoints['paired_regressions']=dict(lower=low,upper=high,maximum=2,total=72,decision='supported' if high<=2 else 'falsified' if low>2 else 'inconclusive')
    return dict(grouped=groups,endpoints=endpoints,paired_changes=pairs)


def accounting_total(audits,identity):
    names=['full_physical_noisy_union','supplied_correlated_deployment','estimated_iid_deployment','shared_between_methods']
    q={key:sum(a['query_accounting'][key] for a in audits) for key in names};assert all(v==1399680 for v in q.values())
    q.update(clean_unique_oracle_cells=identity['clean_unique_oracle_cells'],clean_materialized_cell_instances=524880,source_instances=6480,official_noisy_label_sequence_before=19560279,official_noisy_label_sequence_after=20959959)
    return q


def empty_grid():
    return dict(objective_twice=None,curvature_twice=None,maximum_curvature_tie_degrees=None,selected_degree=None,
                certified_grid_LPs=0,intermediate_fractional_LPs=0,unresolved_degrees=list(range(1,41)),mutations_rejected=0,analytic_zero_proof=False)


def repeat_diagnostics(seed,p,queries,mask):
    # This diagnostic is invoked only by the auditor after both method grids exist.
    shared=np.random.default_rng(p['shared_branch_seed_base']+seed).random((90,81))<float(Fraction(*RHO))
    first,left,right=[queries[name] for name in ['first','repeat1','repeat2']]
    assert (first[shared&mask]==left[shared&mask]).all() and (first[shared&mask]==right[shared&mask]).all()
    disagreements=[int((a[mask]!=b[mask]).sum()) for a,b in [(first,left),(first,right),(left,right)]]
    count=int((shared&mask).sum());observed=int(mask.sum());assert observed==6480
    return dict(shared_branch_observed_positions=count,independent_branch_observed_positions=observed-count,
        physical_query_responses_on_shared_branch=3*count,unanimous_observed_positions=int(((first==left)&(first==right)&mask).sum()),
        pair_disagreements=disagreements,total_pair_disagreements=sum(disagreements),total_pair_comparisons=3*observed,
        nominal_expected_disagreement=[1839,25600],branch_information_supplied_to_either_method=False)


def audit_bank(seed,public=False,protocol=None,references=None):
    p=verify(public=public) if protocol is None else protocol;base=PUBLIC if public else OUT;directory=base/f's{seed}'
    seal=read(base/'query-record-seal.json');assert seal['protocol_sha256']==sha(PROTOCOL) and seal['all72_query_records_validated_before_scoring']
    assert seal['physical_noisy_queries']==1399680 and seal['source_audit_sha256']==sha(base/'source-audit.json')
    assert seal['record_sha256'][str(seed)]==sha(directory/'query-records.npz') and seal['audit_sha256'][str(seed)]==sha(directory/'query-record-audit.json')
    queries=npz(directory/'query-records.npz');mask=npz(directory/'masks.npz')['balanced']
    assert mask.dtype==bool and mask.shape==(90,81) and np.array_equal(mask,reference_masks(p['balanced_seed_base']+seed))
    assert (mask.sum(0)==80).all() and (mask.sum(1)==72).all() and np.array_equal(queries['given'],mask)
    assert set(queries)=={'given','first','repeat1','repeat2'}
    for name in ['first','repeat1','repeat2']:
        v=queries[name];assert v.dtype==np.int8 and v.shape==(90,81) and ((v[mask]>=0)&(v[mask]<=8)).all() and (v[~mask]==-1).all()
    predraw=read(directory/'predraw-receipt.json');assert predraw['seed']==seed and predraw['protocol_sha256']==sha(PROTOCOL) and predraw['masks_sha256']==sha(directory/'masks.npz')
    qa=read(directory/'query-record-audit.json');assert qa['status']=='validated' and qa['seed']==seed and qa['protocol_sha256']==sha(PROTOCOL)
    assert qa['query_records_sha256']==sha(directory/'query-records.npz') and qa['masks_sha256']==sha(directory/'masks.npz')
    assert qa['auditor_sha256']==sha(HERE/'correlated_repeat_data.py')
    assert qa['fresh_raw_correspondence_verified'] and qa['new_queries']==19440 and not qa['scorer_receives_clean_values'] and qa['complete_label_substitution']
    assert qa['channel']==channel_event(RATE)==p['acquisition_channel']
    assert qa['shared_branch_event']==channel_event(RHO)==p['shared_branch_event'] and p['supplied_parameters']==list(PARAMETERS)
    if not public:
        assert qa['noise_sha256']==sha(directory/'potential_noise.npz')
        entry=read(OUT/'input_manifest.json')['inputs'][seed];assert entry['completion_sha256']==qa['completion_sha256']==sha(ROOT/entry['completion_path'])
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));rows=[];primals=[];costs=[]
    for ai,arm in enumerate(ARMS):
        index=2*seed+ai;values,given=arm_values(queries)
        arrays=npz(directory/f'{arm}-models.npz');receipt=read(directory/f'{arm}-receipt.json')
        assert receipt['index']==index and receipt['seed']==seed and receipt['arm']==arm and receipt['protocol_sha256']==sha(PROTOCOL)
        assert receipt['models_sha256']==sha(directory/f'{arm}-models.npz')
        assert receipt['query_records_sha256']==sha(directory/'query-records.npz') and receipt['query_seal_sha256']==sha(base/'query-record-seal.json')
        assert receipt['predraw_receipt_sha256']==sha(directory/'predraw-receipt.json')
        if arm=='estimated_iid':
            estimator=independent_estimate(values,given);assert estimator==estimate(values,given)==receipt['estimator'] and estimator['total_comparisons']==19440
            allowed=estimator['fit_allowed'];channel=(estimator['rate_numerator'],estimator['rate_denominator']) if allowed else None
            producer,reference_fn=estimated_score,estimated_reference
        else:
            estimator=None;assert receipt['estimator'] is None;allowed=True;channel=PARAMETERS;producer,reference_fn=supplied_score,supplied_reference
        numeric=None;widths=[]
        if allowed:
            reproduced=producer(values,given,channel)
            for key,value in reproduced.items():assert np.array_equal(value,arrays[key]),key
            assert (arrays['joint_count']==71).all()
            if references is None:
                try:reference=dict(certified=True,output=reference_fn(values,given,channel),error=None)
                except UncertifiedCost as error:reference=dict(certified=False,output=None,error=str(error))
            else:reference=references[arm]
            numeric=numeric_state(arrays,reference);assert numeric==receipt['numeric']
            if reference['certified']:
                expected,joint,bounds=reference['output'];assert np.array_equal(joint,arrays['joint_count'])
                assert np.array_equal(expected[arrays['cost_certified']],arrays['cost'][arrays['cost_certified']])
                for edge,(lo,hi,scale) in enumerate(bounds):
                    lower,upper=[Decimal(arrays[key][edge]) for key in ['cost_lower','cost_upper']]
                    assert Fraction(lower)<=Fraction(lo*1000000,scale)<=Fraction(hi*1000000,scale)<=Fraction(upper)
                    rounded=rounded_decimal_interval(lower,upper)
                    assert bool(arrays['cost_certified'][edge])==(rounded is not None)
                    assert arrays['cost'][edge]==(0 if rounded is None else rounded)
                    widths.append(Fraction(upper)-Fraction(lower))
        else:
            assert receipt['numeric'] is None and set(arrays)=={'estimator_unexecuted'}
            assert arrays['estimator_unexecuted'].shape==() and arrays['estimator_unexecuted'].dtype==bool and arrays['estimator_unexecuted'].item()
        fit_allowed=allowed and numeric['fit_allowed']
        if fit_allowed:
            assert receipt['unexecuted_grid_reason'] is None and len(receipt['records'])==40
            cost=arrays['cost'];checked=grid(arrays,receipt['records'],cost,p['solver'],index)
            if not public:
                for record in receipt['records']:assert record['log_sha256']==sha(directory/f'{arm}-d{record["degree"]:02d}.log')
            degree=checked['selected_degree'];primal=np.empty(0,np.int64) if degree is None else arrays[f'd{degree:02d}_primal_twice']
            result=statistics(primal,truth)
            fixed=statistics(arrays['d20_primal_twice'],truth) if receipt['records'][19]['certificate']['certified'] else statistics(np.empty(0,np.int64),truth)
            top=np.zeros(3240,bool);top[[e for _,e in sorted((int(cost[e]),e) for e in range(3240))[:810]]]=True
            assert np.array_equal(top,arrays['top810_support']);topstats=statistics(2*top.astype(np.int64),truth)
            gap=None
            if degree==20:
                gap=2*sum(int(c) for c in cost[truth])-checked['objective_twice'][20];assert gap>=0
        else:
            assert receipt['records']==[] and receipt['unexecuted_grid_reason']==('inadmissible_channel_estimate' if not allowed else 'uncertified_producer_or_independent_log_cost_interval')
            assert not any(key.startswith('d') and key[1:3].isdigit() for key in arrays) and 'top810_support' not in arrays
            checked=empty_grid();result=fixed=topstats=statistics(np.empty(0,np.int64),truth);gap=None;primal=np.empty(0,np.int64)
        primals.append(primal);costs.append(arrays['cost'] if fit_allowed else None)
        rows.append(dict(index=index,seed=seed,arm=arm,acquisition_nominal=list(RATE),shared_branch_nominal=list(RHO),estimator=estimator,numeric=numeric,
            **checked,**result,attempted_LPs=len(receipt['records']),fixed_degree20=fixed,top810=topstats,
            truth_gap_twice_if_selected_degree20=gap,max_cost_interval_width=str(max(widths)) if widths else None))
    repeat=repeat_diagnostics(seed,p,queries,mask)
    assert rows[1]['estimator']['disagreements']==repeat['total_pair_disagreements']
    differences=dict(integer_cost_entries=None if any(x is None for x in costs) else int(np.count_nonzero(costs[0]!=costs[1])),
        selected_primal_entries=None if any(not x.size for x in primals) else int(np.count_nonzero(primals[0]!=primals[1])))
    files=['masks.npz','predraw-receipt.json','query-records.npz','query-record-audit.json']+[f'{arm}-{suffix}' for arm in ARMS for suffix in ['models.npz','receipt.json']]
    return dict(status='validated',seed=seed,protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),rows=rows,method_differences=differences,query_accounting=literal_account(mask),
                realized_noise=qa['realized_noise'],channel=qa['channel'],shared_branch_event=qa['shared_branch_event'],repeat_diagnostics=repeat,
                artifact_sha256={name:sha(directory/name) for name in files})


def public_job(seed):return audit_bank(seed,True)


def main():
    started=time.perf_counter();verify(public=True);saved=read(PUBLIC/'summary.json');assert saved['protocol_sha256']==sha(PROTOCOL)
    identity,_,_=identity_audit(True)
    assert saved['source_audit']['auditor_sha256']==sha(HERE/'correlated_repeat_data.py')
    for key,value in identity.items():assert saved['source_audit'][key]==value
    for rel,h in saved['artifact_sha256'].items():assert sha(PUBLIC/rel)==h
    with ProcessPoolExecutor(max_workers=8) as pool:audits=list(pool.map(public_job,range(72)))
    for seed,audit in enumerate(audits):assert audit==read(PUBLIC/f's{seed}/audit.json')
    rows=[r for a in audits for r in a['rows']];assert rows==saved['rows'];outcomes=aggregate(rows)
    for key,value in outcomes.items():assert saved[key]==value
    assert saved['query_accounting']==accounting_total(audits,identity)
    assert saved['attempted_LPs']==sum(r['attempted_LPs'] for r in rows)<=5760
    assert saved['new_certified_LPs']==sum(r['certified_grid_LPs'] for r in rows) and saved['mutations_rejected']==sum(r['mutations_rejected'] for r in rows)
    assert saved['analytic_zero_proofs']==sum(r['analytic_zero_proof'] for r in rows)<=144 and saved['all144_arms_retained']
    print(json.dumps(dict(status='passed',**outcomes,attempted_LPs=saved['attempted_LPs'],new_certified_LPs=saved['new_certified_LPs'],mutations_rejected=saved['mutations_rejected'],source_identity=identity,wall_seconds=time.perf_counter()-started,scope='Public IDs/masks/observations/estimates/costs/proofs replayed; CSV and latent-noise correspondence inherited from charged raw audit.')))


if __name__=='__main__':main()
