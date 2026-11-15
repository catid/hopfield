"""Independent decimal thresholds, conditional multiplier algebra and replay."""
from decimal import Decimal,localcontext
from fractions import Fraction
import hashlib,json
import numpy as np
from null_replay_study_corrected import ROOT,HERE,PARENT,OUT,PROTOCOL,sha,save,verify

def thresholds_audit():
    p=verify();z=np.load(OUT/'thresholds.npz',allow_pickle=False);rows=[];actual=Fraction(9,1600)
    with localcontext()as ctx:
        ctx.prec=80;logbound=Decimal(648000).ln()
        for arm,q0text in p['nulls'].items():
            q0=Fraction(q0text);q1=Fraction(1,9);a=q1/q0;b=(1-q1)/(1-q0);ta=z[arm]
            assert ta.shape==(1111,)and ta.dtype==np.int32
            la=(Decimal(a.numerator)/a.denominator).ln();lb=(Decimal(b.numerator)/b.denominator).ln()
            for m,c in enumerate(ta):
                c=int(c)
                if c<=m:assert c*la+(m-c)*lb>=logbound
                if c>0:assert (c-1)*la+(m-c+1)*lb<logbound
            expectation=actual*a+(1-actual)*b
            if arm=='exact_law':assert expectation==1
            elif arm=='nominal_iid':assert expectation<1 and q0==Fraction(1,20)/4-Fraction(9,64)*Fraction(1,20)**2
            else:assert expectation>1
            rows.append(dict(arm=arm,q0=str(q0),true_pair_equality=str(actual),conditional_multiplier_expectation=str(expectation),expectation_float=float(expectation)))
    result=dict(status='validated',protocol_sha256=sha(PROTOCOL),thresholds_sha256=sha(OUT/'thresholds.npz'),auditor_sha256=sha(__file__),decimal_endpoint_predecessor_checks=3333,conditional_expectations=rows,
        conditional_argument='Given all previous observations and the clean current board, the mask is fixed before the independent event/permutation. For any queried true peer pair, equality has probability9/1600; an unqueried pair has multiplier1. Thus E[L_next/L_past | history,current fixed mask]=p*a+(1-p)*b. This equals1 for exact_law and is<1 for nominal_iid; statements apply only to the supplied independent board-permutation law and predictable mask policy.')
    save(OUT/'threshold-audit.json',result);return result

def audit_bank(seed):
    p=verify();threshold_check=json.loads((OUT/'threshold-audit.json').read_text());assert threshold_check['thresholds_sha256']==sha(OUT/'thresholds.npz')and threshold_check['auditor_sha256']==sha(__file__)
    arms=list(p['nulls']);threshold=np.stack([np.load(OUT/'thresholds.npz')[arm]for arm in arms])
    d=PARENT/f's{seed}';receipt=json.loads((d/'receipt.json').read_text())
    for name in ['observations.npz','fit.npz','calibration.json']:assert receipt['artifact_sha256'][name]==sha(d/name)
    fit=json.loads((d/'calibration.json').read_text())['fit'];eta=Fraction(fit['eta_upper_numerator'],fit['eta_upper_denominator'])
    assert eta/4-Fraction(9,64)*eta**2==Fraction(p['nulls']['calibrated'])
    obs=np.load(d/'observations.npz',allow_pickle=False);given,values=obs['given'],obs['values'];i,j=np.triu_indices(81,1)
    joint=np.zeros(3240,np.int64);equal=np.zeros(3240,np.int64);first=np.full((3,3240),-1,np.int32)
    produced=np.load(OUT/f's{seed}.npz',allow_pickle=False)
    for row in range(1110):
        mask=given[row];assert mask.sum()==25 and(values[row,~mask]==-1).all()
        queried=np.logical_and(mask[i],mask[j]);joint+=queried;equal+=queried&(values[row,i]==values[row,j])
        digest=hashlib.sha256(joint.astype(np.int32).tobytes()+equal.astype(np.int32).tobytes()).hexdigest();assert str(produced['count_sha256'][row])==digest
        for index in range(3):first[index,(first[index]<0)&(equal>=threshold[index,joint])]=row+1
    assert np.array_equal(first,produced['first_crossing'])and np.array_equal(joint,produced['joint_count'])and np.array_equal(equal,produced['equal_count'])
    support=first<0;final=equal[None,:]<threshold[:,joint]
    assert np.array_equal(support,produced['support'])and np.array_equal(final,produced['final_only_support'])
    assert support[:,joint==0].all()and final[:,joint==0].all()and np.all(~support|final)
    original=np.load(d/'fit.npz',allow_pickle=False)
    assert np.array_equal(first[0],original['first_crossing'])and np.array_equal(support[0],original['support'])
    assert np.array_equal(joint,original['joint_count'])and np.array_equal(equal,original['equal_count'])
    truth=(i//9==j//9)|(i%9==j%9)|((i//27==j//27)&(i%9//3==j%9//3));rows=[]
    for index,arm in enumerate(arms):
        missing=int((truth&~support[index]).sum());extra=int((~truth&support[index]).sum())
        rows.append(dict(seed=seed,arm=arm,missing_true_edges=missing,extra_edges=extra,all_true_retained=missing==0,exact_graph=missing==extra==0,
            unexposed_pairs=int((joint==0).sum()),final_only_reinstated_pairs=int((~support[index]&final[index]).sum()),final_only_missing_true=int((truth&~final[index]).sum()),final_only_extras=int((~truth&final[index]).sum()),support_bytes_sha256=hashlib.sha256(support[index].tobytes()).hexdigest()))
    result=dict(status='validated',seed=seed,protocol_sha256=sha(PROTOCOL),auditor_sha256=sha(__file__),model_sha256=sha(OUT/f's{seed}.npz'),rows=rows,
        cumulative_pair_states_checked=3596400,calibrated_first_crossing_paths_matched=3240,parent_receipt_sha256=sha(d/'receipt.json'))
    save(OUT/f's{seed}-audit.json',result);return result
