"""Independent H1cc exact-product log costs, inherited controls and full LP grids."""
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
from fractions import Fraction
import json,time
import numpy as np
from pair_likelihood_study import ROOT,HERE,PARENT,OUT,PUBLIC,PROTOCOL,ARMS,CONTROL,EXPOSURE,read,sha,npz,verify,arm_queries,numeric_state,independent_numeric
from pair_likelihood_math import score,rounded_decimal_interval
from audit_posterior_fresh_confirm import score_reference as parent_score
from audit_degree_prior import check_certificate
from audit_partial_degree import grid
from audit_extra_coverage import statistics


def aggregate(rows):
    assert [r['index'] for r in rows]==list(range(128)) and [r['arm'] for r in rows]==ARMS*64
    groups=[];paired={}
    for ai,arm in enumerate(ARMS):
        part=[r for r in rows if r['arm']==arm]
        assert len(part)==64 and [r['seed'] for r in part]==list(range(64))
        assert sum(r['old_exact'] for r in part)==(64 if ai==0 else 61)
        groups.append(dict(arm=arm,old_exact=sum(r['old_exact'] for r in part),
                           **{kind:sum(r['classification']==kind for r in part) for kind in ['success','resolved_failure_nonexact','resolved_failure_fractional','unresolved']},
                           selected_degree20=sum(r['selected_degree']==20 for r in part),
                           selected_degree_histogram={str(d):sum(r['selected_degree']==d for r in part) for d in range(1,40) if any(r['selected_degree']==d for r in part)},
                           fixed20_exact=sum(r['fixed_degree20']['classification']=='success' for r in part),top810_exact=sum(r['top810']['classification']=='success' for r in part),
                           numerical_unresolved=sum(not r['numeric']['fit_allowed'] for r in part)))
        paired[arm]=dict(gains=[r['seed'] for r in part if not r['old_exact'] and r['classification']=='success'],
                         losses=[r['seed'] for r in part if r['old_exact'] and r['classification'].startswith('resolved_failure')],
                         unresolved_old_success=[r['seed'] for r in part if r['old_exact'] and r['classification']=='unresolved'],
                         both_exact=[r['seed'] for r in part if r['old_exact'] and r['classification']=='success'],
                         both_resolved_failure=[r['seed'] for r in part if not r['old_exact'] and r['classification'].startswith('resolved_failure')],
                         any_unresolved=[r['seed'] for r in part if r['classification']=='unresolved'])
    triple=groups[1];lo=triple['success'];hi=lo+triple['unresolved'];change=paired[ARMS[1]]
    losslo=len(change['losses']);losshi=losslo+len(change['unresolved_old_success'])
    endpoints=dict(triple_exact=dict(lower=lo,upper=hi,required=63,total=64,decision='supported' if lo>=63 else 'falsified' if hi<63 else 'inconclusive'),
                   posterior_regressions=dict(lower=losslo,upper=losshi,maximum=0,total=61,decision='supported' if losshi==0 else 'falsified' if losslo>0 else 'inconclusive'))
    return dict(grouped=groups,paired_changes=paired,endpoints=endpoints)


def empty_grid():
    return dict(objective_twice=None,curvature_twice=None,maximum_curvature_tie_degrees=None,selected_degree=None,
                certified_grid_LPs=0,intermediate_fractional_LPs=0,unresolved_degrees=list(range(1,41)),mutations_rejected=0,analytic_zero_proof=False)


def audit_bank(seed,public=False,protocol=None,references=None):
    p=verify() if protocol is None else protocol;directory=(PUBLIC if public else OUT)/f's{seed}'
    seal=read(PARENT/'query-record-seal.json')
    assert seal['record_sha256'][str(seed)]==sha(PARENT/f's{seed}/query-records.npz') and seal['audit_sha256'][str(seed)]==sha(PARENT/f's{seed}/query-record-audit.json')
    queries=npz(PARENT/f's{seed}/query-records.npz');assert np.array_equal(queries['given'],npz(PARENT/f's{seed}/masks.npz')['balanced'])
    prior=read(PARENT/'summary.json')['rows'];ii,jj=np.triu_indices(81,1)
    truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));rows=[]
    for ai,arm in enumerate(ARMS):
        index=2*seed+ai;values,given=arm_queries(queries,arm)
        arrays=npz(directory/f'{arm}-models.npz');receipt=read(directory/f'{arm}-receipt.json')
        assert receipt['index']==index and receipt['seed']==seed and receipt['arm']==arm and receipt['protocol_sha256']==sha(PROTOCOL)
        assert receipt['models_sha256']==sha(directory/f'{arm}-models.npz')
        assert receipt['parent_query_records_sha256']==sha(PARENT/f's{seed}/query-records.npz') and receipt['parent_query_seal_sha256']==sha(PARENT/'query-record-seal.json')
        reproduced=score(values,given)
        for key,value in reproduced.items():assert np.array_equal(value,arrays[key]),key
        assert (arrays['joint_count']==EXPOSURE[arm]).all() and (given.sum(0)==(240 if ai==0 else 80)).all()
        reference=independent_numeric(values,given) if references is None else references[arm]
        numeric=numeric_state(arrays,reference);assert numeric==receipt['numeric']
        widths=[]
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
        oldarm=CONTROL[arm];oldindex=3*seed+(0 if ai==0 else 2);oldrow=prior[oldindex]
        assert oldrow['seed']==seed and oldrow['arm']==oldarm and oldrow['selected_degree']==20
        oldreceipt=read(PARENT/f's{seed}/{oldarm}-receipt.json');oldpath=PARENT/f's{seed}/{oldarm}-models.npz'
        assert oldreceipt['index']==oldindex and oldreceipt['models_sha256']==sha(oldpath) and oldreceipt['protocol_sha256']==sha(HERE/'protocol-posterior-fresh-confirm.json')
        oldcost,oldjoint,_=parent_score(queries,oldarm)
        with np.load(oldpath,allow_pickle=False) as z:
            assert np.array_equal(oldcost,z['cost']) and np.array_equal(oldjoint,z['joint_count'])
            oldproof=[z['d20_'+key] for key in ['primal_twice','node_dual_twice','upper_dual_twice']]
        assert check_certificate(81,20,ii,jj,oldcost,*oldproof)
        old_exact=bool(np.array_equal(oldproof[0],2*truth.astype(np.int64)));assert old_exact==(oldrow['classification']=='success')
        for which in range(3):
            altered=[v.copy() for v in oldproof];altered[which][0]+=1
            assert not check_certificate(81,20,ii,jj,oldcost,*altered)
        if numeric['fit_allowed']:
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
            checked=empty_grid();result=fixed=topstats=statistics(np.empty(0,np.int64),truth);gap=None
        rows.append(dict(index=index,seed=seed,arm=arm,old_exact=old_exact,archived_control=oldarm,numeric=numeric,
                         **checked,**result,attempted_LPs=len(receipt['records']),fixed_degree20=fixed,top810=topstats,
                         truth_gap_twice_if_selected_degree20=gap,max_cost_interval_width=str(max(widths)) if widths else None))
    files=[f'{arm}-{suffix}' for arm in ARMS for suffix in ['models.npz','receipt.json']]
    return dict(status='validated',seed=seed,protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),rows=rows,
                artifact_sha256={name:sha(directory/name) for name in files})


def public_job(seed):return audit_bank(seed,True)


def main():
    start=time.perf_counter();verify();saved=read(PUBLIC/'summary.json')
    assert saved['protocol_sha256']==sha(PROTOCOL)
    for rel,digest in saved['artifact_sha256'].items():assert sha(PUBLIC/rel)==digest
    with ProcessPoolExecutor(max_workers=8) as pool:audits=list(pool.map(public_job,range(64)))
    for seed,audit in enumerate(audits):assert audit==read(PUBLIC/f's{seed}/audit.json')
    rows=[row for audit in audits for row in audit['rows']];assert rows==saved['rows']
    outcomes=aggregate(rows)
    for key,value in outcomes.items():assert saved[key]==value
    assert saved['attempted_LPs']==sum(r['attempted_LPs'] for r in rows)<=5120
    assert saved['new_certified_LPs']==sum(r['certified_grid_LPs'] for r in rows)
    assert saved['mutations_rejected']==sum(r['mutations_rejected']+3 for r in rows)
    assert saved['analytic_zero_proofs']==sum(r['analytic_zero_proof'] for r in rows)<=128
    assert saved['archived_selected_certificates']==128 and saved['all128_arms_retained']
    assert saved['new_source_IDs']==saved['new_noisy_queries']==0 and saved['parent_physical_noisy_union']==2073600 and saved['official_noisy_label_sequence']==16760919
    print(json.dumps(dict(status='passed',**outcomes,attempted_LPs=saved['attempted_LPs'],new_certified_LPs=saved['new_certified_LPs'],
                         archived_selected_certificates=128,mutations_rejected=saved['mutations_rejected'],wall_seconds=time.perf_counter()-start)))


if __name__=='__main__':main()
