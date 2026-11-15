"""Independent noisy-mask replay, potential-label audit and log-LR paths."""
import argparse
import hashlib
import json
import math
from fractions import Fraction
import time
import numpy as np
from noise_bound_common import load_protocol
from run import ROOT,HERE,sha,save,now

OUT=ROOT/'artifacts/research/20261114/ripper/noise_bound'


def reference_mask(arm,joint,excluded,order):
    if arm=='uniform_e':chosen=list(order[:25])
    else:
        w=(8./9.)**joint.astype(np.float64);w[excluded]=0.;w[np.diag_indices(81)]=0.
        chosen=[]
        for step in range(25):
            score=w.sum(1) if not chosen else np.zeros(81)
            for cell in chosen:score=score+w[:,cell]
            score[chosen]=-np.inf;best=score.max()
            chosen.append(next(int(c) for c in order if score[c]==best))
    mask=np.zeros(81,bool);mask[chosen]=True;return mask


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',required=True,type=int);args=ap.parse_args();start=time.monotonic()
    assert np.__version__=='2.3.4';pp=HERE/'protocol-noise-bound.json';p=load_protocol();d=OUT/f's{args.seed}'
    receipt=json.loads((d/'receipt.json').read_text());assert receipt['protocol_sha256']==sha(pp)
    assert receipt['wrapper_sha256']==sha(HERE/'noise_bound_acquire.py') and receipt['policy_sha256']==sha(HERE/'noise_bound_policy.py')==p['policy_sha256']
    assert receipt['masks_fixed_before_all_reveals']==4500
    manifest=json.loads((ROOT/p['inputs']).read_text());assert sha(ROOT/p['inputs'])==p['input_manifest_sha256']
    entry=next(e for e in manifest['inputs'] if e['seed']==args.seed);source=ROOT/entry['completion_path']
    assert sha(source)==entry['completion_sha256']==receipt['source_sha256'];clean=np.load(source)['solutions']
    preflight_path=ROOT/'artifacts/research/20261114/ripper/noise_bound_preflight.json'
    assert sha(preflight_path)==receipt['preflight_sha256'];preflight=json.loads(preflight_path.read_text());threshold={a:np.array(v) for a,v in preflight['thresholds'].items()}
    assert preflight['exact_control_thresholds_unchanged'] and preflight['decimal_threshold_checks']==4503
    assert sha(d/'potential_noise.npz')==receipt['potential_noise_sha256'];noise=np.load(d/'potential_noise.npz')
    assert sha(d/'events.json')==receipt['events_sha256'];events=json.loads((d/'events.json').read_text());assert len(events)==4500
    archives={}
    for entry in receipt['observations']:
        path=ROOT/entry['path'];assert sha(path)==entry['sha256'];z=np.load(path);arm=entry['arm']
        archives[arm]={k:z[k] for k in z.files};v=archives[arm]
        assert np.all(v['given'].sum(1)==25) and np.all(v['values'][~v['given']]==-1)
    pair=np.triu_indices(81,1);ii,jj=pair
    states={a:{'joint':np.zeros((81,81),np.int32),'equal':np.zeros((81,81),np.int32),'excluded':np.zeros((81,81),bool),
        'first_crossing':np.full(3240,-1,np.int32),'first_excluded':np.full(3240,-1,np.int32)} for a in p['arms']}
    tie_rng=np.random.default_rng(20273000+args.seed);noise_rng=np.random.default_rng(20274000+args.seed)
    mask_checks=0;log_checks=0;models=[]
    for row in range(1500):
        order=tie_rng.permutation(81)
        masks={}
        for ai,arm in enumerate(p['arms']):
            state=states[arm];event=events[row*3+ai]
            assert event['row']==row and event['arm']==arm and event['finalized_before_all_current_reveals']
            before=hashlib.sha256(state['joint'].tobytes()+state['excluded'].tobytes()).hexdigest();assert before==event['past_state_sha256']
            mask=reference_mask(arm,state['joint'],state['excluded'],order);assert np.array_equal(mask,archives[arm]['given'][row])
            assert hashlib.sha256(mask.tobytes()).hexdigest()==event['mask_sha256'];masks[arm]=mask;mask_checks+=1
        # Independently regenerate current potentials only after all masks match.
        flip=noise_rng.random(81)<.01;offset=noise_rng.integers(1,9,81,dtype=np.int8)
        assert np.array_equal(flip,noise['flips'][row]) and np.array_equal(offset,noise['offsets'][row])
        expected=(clean[row]+flip*offset)%9
        for arm,state in states.items():
            mask=masks[arm];values=archives[arm]['values'][row]
            assert np.array_equal(values[mask],expected[mask]);state['joint']+=(mask[:,None]&mask[None,:]).astype(np.int32)
            for digit in range(9):
                cells=np.flatnonzero(mask&(values==digit));state['equal'][np.ix_(cells,cells)]+=1
            m=state['joint'][pair];c=state['equal'][pair]
            eta=Fraction(p['assumed_eta'][arm]);null=eta/4-9*eta**2/64;alternative=Fraction(1,9)
            la=math.log(float(alternative/null));lb=math.log(float((1-alternative)/(1-null)));log=m*lb+c*(la-lb)
            assert np.allclose(log,archives[arm]['log_lr'][row],rtol=0,atol=2e-12);log_checks+=len(log)
            first=state['first_crossing'];first[(first<0)&(c>=threshold[arm][m])]=row+1
            exclude=first>=0
            state['excluded'][pair]=exclude;state['excluded'][(jj,ii)]=exclude
            dropped=state['first_excluded'];dropped[(dropped<0)&exclude]=row+1
            if row+1 in p['sizes']:
                entry=next(x for x in receipt['models'] if x['arm']==arm and x['n']==row+1)
                assert sha(ROOT/entry['path'])==entry['sha256'];z=np.load(ROOT/entry['path'])
                assert np.array_equal(z['support'],~exclude) and np.array_equal(z['joint_count'],m) and np.array_equal(z['equal_count'],c)
                assert np.array_equal(z['first_crossing'],first) and np.array_equal(z['first_excluded'],dropped)
                models.append({'arm':arm,'n':row+1,'support':z['support'].copy(),'m':m.copy()})
    for arm,state in states.items():
        assert np.array_equal(state['first_crossing'],archives[arm]['first_crossing'])
        assert np.array_equal(state['first_excluded'],archives[arm]['first_excluded'])
    # Geometry enters only here, after all masks, counts, paths and supports pass.
    peer=(ii//9==jj//9)|(ii%9==jj%9)|((ii//27==jj//27)&(ii%9//3==jj%9//3));assert peer.sum()==810
    rows=[]
    for m in models:
        support=m['support'];missing=int((peer&~support).sum());extra=int((~peer&support).sum())
        rows.append({'seed':args.seed,'arm':m['arm'],'n':m['n'],'missing_true_edges':missing,'extra_edges':extra,
            'exact_graph':missing==extra==0,'joint_min':int(m['m'].min()),'joint_max':int(m['m'].max()),
            'support_bytes_sha256':hashlib.sha256(support.tobytes()).hexdigest()})
    r={'status':'validated','utc':now(),'seed':args.seed,'protocol_sha256':sha(pp),'wrapper_sha256':sha(__file__),
        'receipt_sha256':sha(d/'receipt.json'),'independent_masks_replayed':mask_checks,'log_lr_values_verified':log_checks,
        'independent_noise_rows_verified':1500,'supports_verified':3,'first_crossing_paths_verified':9720,
        'rows':rows,'elapsed_s':time.monotonic()-start,'numpy':np.__version__}
    save(d/'audit.json',r);print({'seed':args.seed,'mask_checks':mask_checks,'log_checks':log_checks,'elapsed_s':r['elapsed_s']},flush=True)


if __name__=='__main__':main()
