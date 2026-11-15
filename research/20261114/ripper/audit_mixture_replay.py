"""Independent Decimal likelihood boundaries and pair-vector replay for H1at."""
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
import numpy as np
from mixture_replay_study import ROOT,HERE,PARENT,OUT,PROTOCOL,ARMS,sha,save,load_npz,verify


def independent_logs(m,c):
    q0=Fraction(9,1600);alternatives=[Fraction(1,36),Fraction(1,18),Fraction(1,9),Fraction(2,9)]
    logs=[]
    for q in alternatives:
        a=q/q0;b=(1-q)/(1-q0)
        logs.append(c*math.log(float(a))+(m-c)*math.log(float(b)))
    logs=np.array(logs)
    mixture=np.logaddexp(np.logaddexp(logs[0],logs[1]),np.logaddexp(logs[2],logs[3]))-math.log(4)
    return logs,mixture


def audit_thresholds():
    p=verify();threshold=load_npz(OUT/'thresholds.npz')['thresholds']
    assert threshold.shape==(2,1111) and threshold.dtype==np.int32
    expected=[];checks=0;minimum_margin=math.inf
    with localcontext() as ctx:
        ctx.prec=80;bound=Decimal(648000).ln();q0=Decimal(9)/1600
        qs=[Decimal(1)/36,Decimal(1)/18,Decimal(1)/9,Decimal(2)/9]
        log_equal=[(q/q0).ln() for q in qs]
        log_different=[((1-q)/(1-q0)).ln() for q in qs]
        for arm in range(2):
            for m,cut in enumerate(threshold[arm]):
                cut=int(cut);assert 0<cut<=m+1
                for c,must_cross in [(cut-1,False)]+([] if cut>m else [(cut,True)]):
                    terms=[c*a+(m-c)*b for a,b in zip(log_equal,log_different)]
                    peak=max(terms)
                    value=terms[2] if arm==0 else peak+(sum((v-peak).exp() for v in terms)/4).ln()
                    margin=value-bound
                    assert abs(margin)>Decimal('1e-60'),'Near tie requires an explicit independent rational certificate.'
                    assert (margin>=0)==must_cross,(arm,m,c)
                    minimum_margin=min(minimum_margin,float(abs(margin)));checks+=1
        for q in [Fraction(1,36),Fraction(1,18),Fraction(1,9),Fraction(2,9)]:
            null=Fraction(9,1600);expectation=null*(q/null)+(1-null)*((1-q)/(1-null));assert expectation==1
            expected.append(dict(q1=str(q),conditional_multiplier_expectation=str(expectation)))
    a=dict(status='validated',protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),
           thresholds_sha256=sha(OUT/'thresholds.npz'),exposure_states=1111,arms=2,
           decimal_precision=80,decimal_endpoint_predecessor_checks=checks,minimum_log_boundary_margin=minimum_margin,
           conditional_expectations=expected,mixture_weights=['1/4']*4,
           semantics='Both thresholds include equality. All four component likelihoods are unstopped; only the first crossing of their fixed mean is permanent. Independent Decimal80 logsumexp confirms every attainable endpoint and predecessor; no near-boundary rounding ambiguity occurred.')
    save(OUT/'threshold-audit.json',a);return a


def audit_bank(seed):
    p=verify(seed);check=json.loads((OUT/'threshold-audit.json').read_text())
    assert check['status']=='validated' and check['protocol_sha256']==sha(PROTOCOL)
    assert check['auditor_sha256']==sha(__file__) and check['thresholds_sha256']==sha(OUT/'thresholds.npz')
    threshold=load_npz(OUT/'thresholds.npz')['thresholds'];d=PARENT/f's{seed}'
    observed=load_npz(d/'observations.npz');produced=load_npz(OUT/f's{seed}.npz');original=load_npz(d/'fit.npz')
    given,values=observed['given'],observed['values'];archived_log=observed['log_lr']
    assert given.shape==values.shape==(1110,81) and given.dtype==bool and values.dtype==np.int8
    assert archived_log.shape==(1110,3240) and archived_log.dtype==np.float64
    ii,jj=np.triu_indices(81,1);joint=np.zeros(3240,np.int64);equal=np.zeros_like(joint)
    first=np.full((2,3240),-1,np.int32);minimum_margin=math.inf;maximum_original_log_error=0.
    for row in range(1110):
        mask=given[row];assert mask.sum()==25 and (values[row,~mask]==-1).all()
        assert ((values[row,mask]>=0)&(values[row,mask]<9)).all()
        queried=mask[ii]&mask[jj];joint+=queried;equal+=queried&(values[row,ii]==values[row,jj])
        digest=hashlib.sha256(joint.astype(np.int32).tobytes()+equal.astype(np.int32).tobytes()).hexdigest()
        assert produced['count_sha256'][row]==digest
        logs,mix=independent_logs(joint,equal)
        error=float(np.max(np.abs(logs[2]-archived_log[row])))
        maximum_original_log_error=max(maximum_original_log_error,error);assert error<1e-10
        curves=np.stack([logs[2],mix]);direct=curves>=math.log(648000)
        assert np.array_equal(direct,equal[None,:]>=threshold[:,joint])
        minimum_margin=min(minimum_margin,float(np.abs(curves-math.log(648000)).min()))
        first[(first<0)&direct]=row+1
    assert minimum_margin>1e-10
    assert np.array_equal(first,produced['first_crossing'])
    assert np.array_equal(joint,produced['joint_count']) and np.array_equal(equal,produced['equal_count'])
    support=first<0;final=equal[None,:]<threshold[:,joint]
    assert np.array_equal(support,produced['support']) and np.array_equal(final,produced['final_only_support'])
    assert support[:,joint==0].all() and final[:,joint==0].all() and np.all(~support|final)
    assert np.array_equal(first[0],original['first_crossing']) and np.array_equal(support[0],original['support'])
    assert np.array_equal(joint,original['joint_count']) and np.array_equal(equal,original['equal_count'])
    assert np.allclose(logs,produced['final_component_log_lr'],rtol=0,atol=1e-10)
    assert np.allclose(mix,produced['final_mixture_log_lr'],rtol=0,atol=1e-10)
    truth=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));assert truth.sum()==810
    rows=[]
    for index,arm in enumerate(ARMS):
        missing=int((truth&~support[index]).sum());extra=int((~truth&support[index]).sum())
        rows.append(dict(seed=seed,arm=arm,missing_true_edges=missing,extra_edges=extra,
                         all_true_retained=missing==0,exact_graph=missing==extra==0,
                         final_only_reinstated_pairs=int((~support[index]&final[index]).sum()),
                         unexposed_pairs=int((joint==0).sum()),
                         support_bytes_sha256=hashlib.sha256(support[index].tobytes()).hexdigest()))
    result=dict(status='validated',seed=seed,protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),
                model_sha256=sha(OUT/f's{seed}.npz'),rows=rows,cumulative_pair_states_checked=3596400,
                original_first_crossing_paths_matched=3240,mixture_first_crossing_paths_checked=3240,
                minimum_log_boundary_margin=minimum_margin,maximum_original_log_error=maximum_original_log_error,
                parent_observations_sha256=sha(d/'observations.npz'),parent_receipt_sha256=sha(d/'receipt.json'))
    save(OUT/f's{seed}-audit.json',result);return result
