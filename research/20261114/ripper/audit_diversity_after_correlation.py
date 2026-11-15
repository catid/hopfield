"""Independent H1ci fresh observations, fixed-method costs, proofs and endpoints."""
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
from fractions import Fraction
import json,time
import numpy as np
from diversity_after_correlation_study import ROOT,HERE,OUT,PUBLIC,PROTOCOL,ARMS,RATE,RHO,PARAMETERS,read,sha,npz,verify,arm_values,numeric_state
from diversity_after_correlation_data import identity_audit,reference_masks,literal_account,channel_event
from correlated_repeat_math import score as supplied_score,independent_score as supplied_reference
from channel_misspecification_math import score as single_score,independent_score as single_reference
from pair_likelihood_math import rounded_decimal_interval,UncertifiedCost
from audit_partial_degree import grid
from audit_extra_coverage import statistics


def aggregate(rows):
    assert [r['index'] for r in rows]==list(range(144)) and [r['arm'] for r in rows]==ARMS*72
    groups=[]
    for ai,arm in enumerate(ARMS):
        part=rows[ai::2];assert [r['seed'] for r in part]==list(range(72))
        group=dict(arm=arm,**{kind:sum(r['classification']==kind for r in part) for kind in ['success','resolved_failure_nonexact','resolved_failure_fractional','unresolved']},
            selected_degree20=sum(r['selected_degree']==20 for r in part),fixed20_exact=sum(r['fixed_degree20']['classification']=='success' for r in part),top810_exact=sum(r['top810']['classification']=='success' for r in part),
            numerical_unresolved=sum(not r['numeric']['fit_allowed'] for r in part))
        groups.append(group)
    pairs={k:[] for k in ['gains','losses','both_exact','both_resolved_failure','any_unresolved']};low=high=0
    failures=['resolved_failure_nonexact','resolved_failure_fractional']
    for seed in range(72):
        repeated,diverse=[rows[2*seed+ai]['classification'] for ai in [0,1]]
        assert repeated in failures+['success','unresolved'] and diverse in failures+['success','unresolved']
        low+=int(diverse=='success')-int(repeated in ['success','unresolved'])
        high+=int(diverse in ['success','unresolved'])-int(repeated=='success')
        key='any_unresolved' if 'unresolved' in [repeated,diverse] else 'both_exact' if repeated==diverse=='success' else 'gains' if diverse=='success' else 'losses' if repeated=='success' else 'both_resolved_failure'
        pairs[key].append(seed)
    lo=groups[1]['success'];hi=lo+groups[1]['unresolved']
    endpoints=dict(diverse_exact=dict(lower=lo,upper=hi,required=60,total=72,decision='supported' if lo>=60 else 'falsified' if hi<60 else 'inconclusive'),
        net_recovery_gain=dict(lower=low,upper=high,required=48,total=72,decision='supported' if low>=48 else 'falsified' if high<48 else 'inconclusive'))
    return dict(grouped=groups,endpoints=endpoints,paired_changes=pairs)


def accounting_total(audits,identity):
    names=['full_physical_noisy_union','supplied_correlated90_deployment','supplied_single270_deployment','shared_between_methods']
    q={key:sum(a['query_accounting'][key] for a in audits) for key in names}
    assert q==dict(full_physical_noisy_union=2332800,supplied_correlated90_deployment=1399680,supplied_single270_deployment=1399680,shared_between_methods=466560)
    q.update(clean_unique_oracle_cells=identity['clean_unique_oracle_cells'],clean_materialized_cell_instances=1574640,source_instances=19440,
        repeated_deployment_source_instances=6480,diverse_deployment_source_instances=19440,official_noisy_label_sequence_before=20959959,official_noisy_label_sequence_after=23292759)
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
    assert seal['physical_noisy_queries']==2332800 and seal['source_audit_sha256']==sha(base/'source-audit.json')
    assert seal['record_sha256'][str(seed)]==sha(directory/'query-records.npz') and seal['audit_sha256'][str(seed)]==sha(directory/'query-record-audit.json')
    queries=npz(directory/'query-records.npz');mask=npz(directory/'masks.npz')['balanced']
    assert mask.dtype==bool and mask.shape==(270,81) and np.array_equal(mask,np.concatenate([reference_masks(base+seed) for base in p['balanced_seed_bases']]))
    assert (mask.sum(0)==240).all() and (mask.sum(1)==72).all() and np.array_equal(queries['given'],mask)
    assert set(queries)=={'given','first','repeat1','repeat2','diversity1','diversity2'}
    for name,block in [('first',0),('repeat1',0),('repeat2',0),('diversity1',1),('diversity2',2)]:
        given=mask[90*block:90*(block+1)];v=queries[name]
        assert v.dtype==np.int8 and v.shape==(90,81) and ((v[given]>=0)&(v[given]<=8)).all() and (v[~given]==-1).all()
    repeated,rg=arm_values(queries,ARMS[0]);diverse,dg=arm_values(queries,ARMS[1])
    assert np.array_equal(repeated[0],diverse[0,:90]) and np.array_equal(rg,dg[:90])
    predraw=read(directory/'predraw-receipt.json');assert predraw['seed']==seed and predraw['protocol_sha256']==sha(PROTOCOL) and predraw['masks_sha256']==sha(directory/'masks.npz')
    qa=read(directory/'query-record-audit.json');assert qa['status']=='validated' and qa['seed']==seed and qa['protocol_sha256']==sha(PROTOCOL)
    assert qa['query_records_sha256']==sha(directory/'query-records.npz') and qa['masks_sha256']==sha(directory/'masks.npz')
    assert qa['auditor_sha256']==sha(HERE/'diversity_after_correlation_data.py')
    assert qa['fresh_raw_correspondence_verified'] and qa['new_queries']==32400 and qa['literal_shared_first90_response_and_mask'] and not qa['scorer_receives_clean_values'] and qa['complete_label_substitution']
    assert qa['channel']==channel_event(RATE)==p['acquisition_channel']
    assert qa['shared_branch_event']==channel_event(RHO)==p['shared_branch_event'] and p['supplied_parameters']==list(PARAMETERS)
    if not public:
        assert qa['noise_sha256']==sha(directory/'potential_noise.npz')
        entry=read(OUT/'input_manifest.json')['inputs'][seed];assert entry['completion_sha256']==qa['completion_sha256']==sha(ROOT/entry['completion_path'])
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));rows=[];primals=[];costs=[]
    for ai,arm in enumerate(ARMS):
        index=2*seed+ai;values,given=arm_values(queries,arm)
        arrays=npz(directory/f'{arm}-models.npz');receipt=read(directory/f'{arm}-receipt.json')
        assert receipt['index']==index and receipt['seed']==seed and receipt['arm']==arm and receipt['protocol_sha256']==sha(PROTOCOL)
        assert receipt['models_sha256']==sha(directory/f'{arm}-models.npz')
        assert receipt['query_records_sha256']==sha(directory/'query-records.npz') and receipt['query_seal_sha256']==sha(base/'query-record-seal.json')
        assert receipt['predraw_receipt_sha256']==sha(directory/'predraw-receipt.json')
        estimator=None;assert receipt['estimator'] is None
        if arm=='supplied_correlated90':channel=PARAMETERS;producer,reference_fn=supplied_score,supplied_reference
        else:channel=RATE;producer,reference_fn=single_score,single_reference
        numeric=None;widths=[]
        reproduced=producer(values,given,channel)
        for key,value in reproduced.items():assert np.array_equal(value,arrays[key]),key
        assert (arrays['joint_count']==(71 if arm=='supplied_correlated90' else 213)).all()
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
        fit_allowed=numeric['fit_allowed']
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
            assert receipt['records']==[] and receipt['unexecuted_grid_reason']=='uncertified_producer_or_independent_log_cost_interval'
            assert not any(key.startswith('d') and key[1:3].isdigit() for key in arrays) and 'top810_support' not in arrays
            checked=empty_grid();result=fixed=topstats=statistics(np.empty(0,np.int64),truth);gap=None;primal=np.empty(0,np.int64)
        primals.append(primal);costs.append(arrays['cost'] if fit_allowed else None)
        rows.append(dict(index=index,seed=seed,arm=arm,acquisition_nominal=list(RATE),shared_branch_nominal=list(RHO) if arm=='supplied_correlated90' else None,estimator=estimator,numeric=numeric,
            **checked,**result,attempted_LPs=len(receipt['records']),fixed_degree20=fixed,top810=topstats,
            truth_gap_twice_if_selected_degree20=gap,max_cost_interval_width=str(max(widths)) if widths else None))
    repeat=repeat_diagnostics(seed,p,queries,mask[:90])
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
    assert saved['source_audit']['auditor_sha256']==sha(HERE/'diversity_after_correlation_data.py')
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
