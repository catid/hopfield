"""Independent observed-label filter, full adaptive replay, and clean replay."""
import argparse
from collections import Counter
import hashlib
import json
import math
import time
import numpy as np
from run import ROOT,HERE,sha,save,now
from balanced_filter_common import paths,load_protocol
from fast_reference_mask import reference_mask
from audit_correlated_noise_acquisition import increment
from audit_collapsed_noise_acquisition import independent_draw,calibration_audit


def state_hash(joint,equal,excluded):
    return hashlib.sha256(joint.tobytes()+equal.tobytes()+excluded.tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);parser.add_argument('--arm',choices=['unfiltered','filtered'],required=True)
    args=parser.parse_args();start=time.monotonic();assert np.__version__=='2.3.4'
    out,pp=paths();p=load_protocol();seed=args.seed;arm=args.arm;assert seed in p['seeds'];bank=out/f's{seed}';d=bank/arm
    receipt=json.loads((d/'receipt.json').read_text());assert receipt['status']=='complete' and receipt['protocol_sha256']==sha(pp)
    assert receipt['collector_sha256']==sha(HERE/'balanced_filter_acquire.py') and receipt['queries_per_arm']==29750
    assert receipt['training_queries_spent']==27750 and receipt['arm']==arm and receipt['seed']==seed
    for name,digest in receipt['artifact_sha256'].items():assert sha(d/name)==digest
    entry=next(e for e in json.loads((ROOT/p['inputs']).read_text())['inputs'] if e['seed']==seed);source={}
    for role in ('calibration','training'):
        path=ROOT/entry[role+'_path'];assert sha(path)==entry[role+'_sha256']==receipt[role+'_source_sha256'];source[role]=np.load(path)['solutions']
        for board in source[role]:assert Counter(map(int,board))==Counter({value:9 for value in range(9)})
    cal_receipt=json.loads((bank/'calibration_receipt.json').read_text())
    assert receipt['shared_calibration_receipt_sha256']==sha(bank/'calibration_receipt.json')
    assert cal_receipt['protocol_sha256']==sha(pp) and cal_receipt['source_sha256']==entry['calibration_sha256']
    assert cal_receipt['estimator_calls']==1 and cal_receipt['shared_across_both_arms']
    for name,digest in cal_receipt['artifact_sha256'].items():assert sha(bank/name)==digest
    eta,a,b,threshold,fit,cal_errors,cal_events=calibration_audit(bank,p,seed,source['calibration'])
    la,lb=math.log(float(a)),math.log(float(b));z=np.load(d/'observations.npz');obs={k:z[k] for k in z.files}
    z=np.load(d/'potential_noise.npz');noise={k:z[k] for k in z.files}
    for array in noise.values():array.setflags(write=False)
    noise_hash={key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in noise.items()}
    events=json.loads((d/'events.json').read_text());assert len(events)==1110 and np.all(obs['given'].sum(1)==25)
    assert obs['accepted'].dtype==obs['violates_digit_balance'].dtype==bool
    assert np.all(obs['values'][~obs['given']]==-1)
    joint=np.zeros((81,81),np.int32);equal=np.zeros_like(joint);excluded=np.zeros((81,81),bool);first=np.full(3240,-1,np.int32)
    cj=joint.copy();ce=equal.copy();cx=excluded.copy();cf=first.copy()
    pair=np.triu_indices(81,1);i,j=pair;truth=(i//9==j//9)|(i%9==j%9)|((i//27==j//27)&(i%9//3==j%9//3))
    ties=np.random.default_rng(p['streams']['ties']+seed);clean_ties=np.random.default_rng(p['streams']['ties']+seed)
    nrng=np.random.default_rng(p['streams']['training_noise']+seed)
    training_errors=accepted_errors=collapse_events=accepted_corrupt=rejected_clean=accepted_count=clean_rows=0
    true_exposures=true_equal=0
    for row in range(1110):
        order=ties.permutation(81);event=events[row];before=state_hash(joint,equal,excluded);old_first=first.copy()
        assert event['row']==row and event['mask_finalized_before_current_corruption'] and event['filter_uses_only_observed_labels']
        assert event['past_state_sha256']==before
        mask=reference_mask('exposure_exact',joint,excluded,order)
        assert np.array_equal(mask,obs['given'][row]) and event['mask_sha256']==hashlib.sha256(mask.tobytes()).hexdigest()
        # Both replays advance the row-indexed tie stream even on rejected rows.
        clean_order=clean_ties.permutation(81);assert np.array_equal(clean_order,order)
        if arm=='filtered':
            clean_mask=reference_mask('exposure_exact',cj,cx,clean_order);assert np.array_equal(clean_mask,mask)
        # Current corruption is reconstructed only after each mask is fixed.
        uniform,digit,collapse=independent_draw(nrng);collapse_events+=int(collapse)
        assert noise['uniforms'][row]==uniform and noise['digits'][row]==digit
        expected=np.full(81,digit,np.int8) if collapse else source['training'][row]
        values=obs['values'][row];assert np.array_equal(values[mask],expected[mask])
        label_counts=Counter(int(value) for value in values[mask]);violation=max(label_counts.values())>9
        assert violation==collapse==bool(obs['violates_digit_balance'][row])
        accepted=arm=='unfiltered' or not violation
        assert accepted==bool(obs['accepted'][row])==event['accepted'];accepted_count+=int(accepted)
        accepted_corrupt+=int(accepted and collapse);rejected_clean+=int(not accepted and not collapse)
        errors=int((values[mask]!=source['training'][row,mask]).sum());training_errors+=errors
        if accepted:
            accepted_errors+=errors;increment(joint,equal,mask,values)
        m,c=joint[pair],equal[pair]
        assert np.allclose(obs['log_lr'][row],m*lb+c*(la-lb),rtol=0,atol=2e-12)
        first[(first<0)&(c>=threshold[m])]=row+1;drop=first>=0;excluded[pair]=drop;excluded[(pair[1],pair[0])]=drop
        if not accepted:
            assert state_hash(joint,equal,excluded)==before and np.array_equal(first,old_first)
        if arm=='filtered':
            if accepted:
                clean_values=np.full(81,-1,np.int8);clean_values[clean_mask]=source['training'][row,clean_mask]
                increment(cj,ce,clean_mask,clean_values)
            cm,cc=cj[pair],ce[pair];cf[(cf<0)&(cc>=threshold[cm])]=row+1
            clean_drop=cf>=0;cx[pair]=clean_drop;cx[(pair[1],pair[0])]=clean_drop
            assert np.array_equal(joint,cj) and np.array_equal(equal,ce) and np.array_equal(excluded,cx) and np.array_equal(first,cf)
            assert not (cc[truth]).any();clean_rows+=1
        queried=mask[i]&mask[j]
        if accepted:
            true_exposures+=int((queried&truth).sum());true_equal+=int((queried&truth&(values[i]==values[j])).sum())
    archived=np.load(d/'fit.npz');support=first<0
    for key,value in [('support',support),('joint_count',m),('equal_count',c),('first_crossing',first)]:assert np.array_equal(archived[key],value)
    assert true_exposures==int(m[truth].sum()) and true_equal==int(c[truth].sum())
    assert noise_hash=={key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in noise.items()}
    assert receipt['accepted_boards']==accepted_count and receipt['rejected_boards']==1110-accepted_count
    missing=int((truth&~support).sum());extras=int((~truth&support).sum())
    if arm=='filtered':
        assert accepted_corrupt==rejected_clean==accepted_errors==missing==0
        assert accepted_count==1110-collapse_events and clean_rows==1110
    else:assert accepted_count==1110 and accepted_corrupt==collapse_events
    report={'status':'validated','utc':now(),'seed':seed,'arm':arm,'protocol_sha256':sha(pp),'auditor_sha256':sha(__file__),
        'receipt_sha256':sha(d/'receipt.json'),'shared_calibration_receipt_sha256':sha(bank/'calibration_receipt.json'),
        'independent_masks_replayed':1110,'log_lr_values_verified':3596400,'first_crossing_paths_verified':3240,'decimal_threshold_checks':1111,
        'calibration_fit':fit,'calibration_error_queries':cal_errors,'calibration_queries_shared':2000,'calibration_collapse_events':cal_events,
        'training_error_queries':training_errors,'training_queries_spent':27750,'accepted_error_queries':accepted_errors,'accepted_queries':25*accepted_count,
        'training_collapse_events':collapse_events,'accepted_boards':accepted_count,'rejected_boards':1110-accepted_count,
        'accepted_corrupt_boards':accepted_corrupt,'rejected_clean_boards':rejected_clean,'clean_subsequence_trajectory_rows_identical':clean_rows,
        'true_pair_exposures':true_exposures,'true_pair_equal_count':true_equal,'missing_true_edges':missing,'extra_edges':extras,'exact_graph':missing==extras==0,
        'support_bytes_sha256':hashlib.sha256(support.tobytes()).hexdigest(),'elapsed_s':time.monotonic()-start}
    save(d/'audit.json',report);print(json.dumps({'seed':seed,'arm':arm,'missing_true':missing,'extra_edges':extras,'accepted':accepted_count}),flush=True)


if __name__=='__main__':main()
