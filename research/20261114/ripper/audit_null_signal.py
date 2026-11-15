"""Independent null observations, all real/null LP proofs and exact ratio contrast."""
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import hashlib,json,time
import numpy as np
from null_signal_study import ROOT,HERE,OUT,PUBLIC,PARENT,PROTOCOL,ARMS,sha,save,npz,verify
from audit_balanced_mask_math import reference_balanced
from audit_extra_coverage import statistics
from audit_partial_degree import grid
from audit_laplace_rate import independent_rate_costs


def reference_masks(seed):
    p=json.loads((HERE/'protocol-higher-noise.json').read_text());rng=np.random.default_rng(p['rank_seed_base']+seed);uniform=np.zeros((180,81),bool)
    for row in range(180):
        for cell in list(map(int,rng.permutation(81)))[:72]:uniform[row,cell]=True
    return dict(uniform=uniform,balanced=reference_balanced(p['balanced_seed_base']+seed))


def reference_null(seed):return np.random.default_rng(seed).integers(0,9,size=180*81,dtype=np.int8).reshape(180,81)


def reference_observe(given,labels):
    values=np.where(given,labels,np.int8(-1));joint=[];equal=[]
    for left in range(80):
        both=given[:,left,None]&given[:,left+1:];joint.extend(map(int,both.sum(axis=0)));equal.extend(map(int,(both&(labels[:,left,None]==labels[:,left+1:])).sum(axis=0)))
    return dict(given=given,values=values,joint_count=np.array(joint,np.int64),equal_count=np.array(equal,np.int64))


def reference_concentration(values):
    if None in values:return None
    assert len(values)==41;peaks=[]
    for degree in range(1,40):
        left=int(values[degree])-int(values[degree-1]);right=int(values[degree+1])-int(values[degree]);assert right>=left;peaks.append(right-left)
    total=sum(peaks);return dict(numerator=max(peaks) if total else 0,denominator=total if total else 1,all_zero_curvature=not bool(total))


def reference_compare(real,null):
    if real is None or null is None:return None
    return Fraction(real['numerator'],real['denominator'])>Fraction(null['numerator'],null['denominator'])


def literal_account(masks):
    a={(row,int(cell)) for row,mask in enumerate(masks['uniform']) for cell in np.flatnonzero(mask)};b={(row,int(cell)) for row,mask in enumerate(masks['balanced']) for cell in np.flatnonzero(mask)}
    return dict(uniform_queries=len(a),balanced_queries=len(b),shared_queries=len(a&b),new_union_queries=len(a|b))


def label_gate(masks,labels,expected):
    from null_signal_study import observe
    changed=((labels+(np.arange(81)%8+1)[None,:])%9).astype(np.int8);assert (changed!=labels).all();gates=[]
    for arm in ARMS:
        old=expected[arm];actual=observe(masks[arm],changed);given=masks[arm]
        assert np.array_equal(actual['given'],old['given']) and np.array_equal(actual['joint_count'],old['joint_count']) and (actual['values'][given]!=old['values'][given]).all() and not np.array_equal(actual['equal_count'],old['equal_count'])
        gates.append(dict(arm=arm,status='passed',all_generated_and_queried_labels_changed=True,masks_and_joint_counts_unchanged=True,equality_counts_changed=True,original_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),changed_sha256=hashlib.sha256(changed.tobytes()).hexdigest()))
    return gates


def audit_bank(seed,public=False):
    from null_signal_study import concentration,strict_greater
    p=verify();base=PUBLIC if public else OUT;d=base/f's{seed}';masks=npz(d/'masks.npz');assert sha(d/'masks.npz')==sha(PARENT/f's{seed}/masks.npz');reference=reference_masks(seed)
    for arm in ARMS:assert np.array_equal(masks[arm],reference[arm]) and (masks[arm].sum(axis=1)==72).all()
    labels=npz(d/'null-labels.npz')['labels'];assert labels.dtype==np.int8 and np.array_equal(labels,reference_null(p['null_seed_base']+seed));expected={arm:reference_observe(masks[arm],labels) for arm in ARMS}
    ii,jj=np.triu_indices(81,1);truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));parent=json.loads((PARENT/'summary.json').read_text());rows=[]
    for ai,arm in enumerate(ARMS):
        index=2*seed+ai;real=npz(PARENT/f's{seed}/{arm}-models.npz');rr=json.loads((PARENT/f's{seed}/{arm}-receipt.json').read_text());prior=parent['rows'][index]
        assert rr['index']==index and rr['seed']==seed and rr['arm']==arm and rr['protocol_sha256']==p['parent_protocol_sha256'] and rr['models_sha256']==sha(PARENT/f's{seed}/{arm}-models.npz')
        assert prior['index']==index and prior['seed']==seed and prior['arm']==arm
        realcost=independent_rate_costs(real['joint_count'],real['equal_count']);assert np.array_equal(realcost,real['cost']) and np.array_equal(real['joint_count'],expected[arm]['joint_count']);realgrid=grid(real,rr['records'],realcost,p['parent_solver'],index)
        for key,value in realgrid.items():assert prior[key]==value
        realout=statistics(real[f'd{realgrid["selected_degree"]:02d}_primal_twice'],truth) if realgrid['selected_degree'] is not None else statistics(np.empty(0,np.int64),truth)
        for key,value in realout.items():assert prior[key]==value
        null=npz(d/f'{arm}-models.npz');receipt=json.loads((d/f'{arm}-receipt.json').read_text());assert receipt['index']==index and receipt['seed']==seed and receipt['arm']==arm and receipt['protocol_sha256']==sha(PROTOCOL) and len(receipt['records'])==40
        assert receipt['models_sha256']==sha(d/f'{arm}-models.npz') and receipt['masks_sha256']==sha(d/'masks.npz') and receipt['null_labels_sha256']==sha(d/'null-labels.npz')
        if not public:
            assert receipt['observations_sha256']==sha(d/f'{arm}-observations.npz');observed=npz(d/f'{arm}-observations.npz')
            for key,value in expected[arm].items():assert np.array_equal(value,observed[key])
            for r in receipt['records']:assert r['log_sha256']==sha(d/f'{arm}-d{r["degree"]:02d}.log')
        for key in ['joint_count','equal_count']:assert np.array_equal(null[key],expected[arm][key])
        cost=independent_rate_costs(null['joint_count'],null['equal_count']);assert np.array_equal(cost,null['cost']);nullgrid=grid(null,receipt['records'],cost,p['solver'],index);degree=nullgrid['selected_degree'];outcome=statistics(np.empty(0,np.int64) if degree is None else null[f'd{degree:02d}_primal_twice'],truth)
        realratio=reference_concentration(realgrid['objective_twice']);nullratio=reference_concentration(nullgrid['objective_twice']);comparison=reference_compare(realratio,nullratio)
        assert realratio==concentration(realgrid['objective_twice']) and nullratio==concentration(nullgrid['objective_twice']) and comparison==strict_greater(realratio,nullratio)
        top=np.zeros(3240,bool);top[[edge for _,edge in sorted((int(cost[e]),e) for e in range(3240))[:810]]]=True;assert np.array_equal(top,null['top810_support'])
        fixed=statistics(null['d20_primal_twice'],truth) if receipt['records'][19]['certificate']['certified'] else statistics(np.empty(0,np.int64),truth)
        rows.append(dict(index=index,seed=seed,arm=arm,real_grid=realgrid,null_grid=nullgrid,real_concentration=realratio,null_concentration=nullratio,real_greater=comparison,real_classification=prior['classification'],null_outcome=outcome,null_fixed20=fixed,null_top810=statistics(2*top.astype(np.int64),truth)))
    shared=masks['uniform']&masks['balanced'];assert np.array_equal(expected['uniform']['values'][shared],expected['balanced']['values'][shared]);gates=label_gate(masks,labels,expected)
    files=['masks.npz','null-labels.npz']+[arm+'-'+suffix for arm in ARMS for suffix in ['models.npz','receipt.json']]
    if not public:files+=[arm+'-observations.npz' for arm in ARMS]
    return dict(status='validated',seed=seed,protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),rows=rows,query_accounting=literal_account(masks),complete_null_label_gates=gates,artifact_sha256={name:sha(d/name) for name in files})


def aggregate(rows):
    assert [r['index'] for r in rows]==list(range(64));grouped=[]
    for arm in ARMS:
        part=[r for r in rows if r['arm']==arm];assert len(part)==32
        grouped.append(dict(arm=arm,real_greater=sum(r['real_greater'] is True for r in part),real_not_greater=sum(r['real_greater'] is False for r in part),unresolved_comparison=sum(r['real_greater'] is None for r in part),
                            null_degree_histogram={str(d):sum(r['null_grid']['selected_degree']==d for r in part) for d in range(1,40) if any(r['null_grid']['selected_degree']==d for r in part)},null_degree20=sum(r['null_grid']['selected_degree']==20 for r in part),
                            **{'null_'+key:sum(r['null_outcome']['classification']==key for r in part) for key in ['success','resolved_failure_nonexact','resolved_failure_fractional','unresolved']},null_fixed20_exact=sum(r['null_fixed20']['classification']=='success' for r in part),null_top810_exact=sum(r['null_top810']['classification']=='success' for r in part)))
    banks=dict(success=[],resolved_failure=[],unresolved=[])
    for seed in range(32):
        outcomes=[rows[2*seed+i]['real_greater'] for i in range(2)];key='resolved_failure' if False in outcomes else 'success' if outcomes==[True,True] else 'unresolved';banks[key].append(seed)
    low=len(banks['success']);high=low+len(banks['unresolved']);endpoint=dict(lower=low,upper=high,required=29,total=32,decision='supported' if low>=29 else 'falsified' if high<29 else 'inconclusive')
    return dict(grouped=grouped,joint_banks=banks,endpoint=endpoint)


def public_job(seed):return audit_bank(seed,True)


def public_audit():
    start=time.monotonic();p=verify();saved=json.loads((PUBLIC/'summary.json').read_text());assert saved['protocol_sha256']==sha(PROTOCOL)
    for name,h in saved['artifact_sha256'].items():assert sha(PUBLIC/name)==h
    with ProcessPoolExecutor(max_workers=8) as pool:audits=list(pool.map(public_job,range(32)))
    rows=[r for a in audits for r in a['rows']];assert rows==saved['rows'];result=aggregate(rows)
    for key,value in result.items():assert saved[key]==value
    q={k:sum(a['query_accounting'][k] for a in audits) for k in audits[0]['query_accounting']};assert q==saved['literal_null_query_accounting'] and saved['query_accounting']==p['query_accounting']
    assert q==dict(uniform_queries=414720,balanced_queries=414720,shared_queries=368719,new_union_queries=460721)
    assert saved['null_certified_LPs']==sum(r['null_grid']['certified_grid_LPs'] for r in rows) and saved['real_certified_LPs_rechecked']==sum(r['real_grid']['certified_grid_LPs'] for r in rows)
    assert saved['mutations_rejected']==sum(r['null_grid']['mutations_rejected']+r['real_grid']['mutations_rejected'] for r in rows)
    assert sum(len(a['complete_null_label_gates']) for a in audits)==saved['complete_null_label_gates']==64
    print(json.dumps(dict(status='passed',**result,null_certified_LPs=saved['null_certified_LPs'],real_certified_LPs_rechecked=saved['real_certified_LPs_rechecked'],mutations_rejected=saved['mutations_rejected'],wall_seconds=time.monotonic()-start,scope='Full synthetic null replay, archived real masks/count costs/proofs, exact ratio contrast and accounting; inherited real-source/noise provenance from H1bq.')))


if __name__=='__main__':public_audit()
