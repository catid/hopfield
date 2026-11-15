"""Independent fresh-bank replay from past queried observations only."""
import argparse
import ast
import hashlib
import json
import time
import numpy as np
from recovery_common import load_protocol
from run import ROOT,HERE,save,sha,now
from clues_estimator import estimate

OUT=ROOT/'artifacts/research/20261114/ripper/recovery'
PROTOCOL=HERE/'protocol-recovery.json'


def reference_mask(arm,joint,equal,order):
    if arm=='uniform':chosen=list(order[:25])
    else:
        if arm in ('spike','exponential'):
            u=np.power(8.0/9.0,joint.astype(np.float64))
            w=np.where(equal==0,2*u/(1+u) if arm=='spike' else u,0.)
        else:
            assert arm=='candidate'
            w=np.reciprocal(joint.astype(np.float64)+1)
            w=np.where(equal==0,w,0.)
        w[np.diag_indices(81)]=0.
        chosen=[]
        for step in range(25):
            if step==0:score=np.sum(w,axis=1)
            else:
                score=np.zeros(81)
                for previous in chosen:score=score+w[:,previous]
            score[chosen]=-np.inf
            maximum=np.max(score)
            chosen.append(next(int(cell) for cell in order if score[cell]==maximum))
    out=np.zeros(81,bool);out[chosen]=True;return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);a=ap.parse_args()
    start=time.monotonic();p=load_protocol();d=OUT/f's{a.seed}'
    r=json.loads((d/'receipt.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL) and r['wrapper_sha256']==sha(HERE/'recovery_acquire.py')
    assert r['policy_sha256']==sha(HERE/'exponential_policy.py') and r['current_row_permutation_invariance_checks']==3000
    assert r['premature_reveal_rejections']==3000
    tree=ast.parse((HERE/'exponential_policy.py').read_text())
    imports=[n for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert len(imports)==2 and isinstance(imports[0],ast.Import) and imports[0].names[0].name=='numpy'
    assert isinstance(imports[1],ast.ImportFrom) and imports[1].module=='spike_policy'
    assert sha(HERE/'online_policy.py')==p['original_policy_sha256']
    assert sha(HERE/'spike_policy.py')==p['mixture_policy_sha256']
    choose=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='choose_mask')
    assert [v.arg for v in choose.args.args]==['arm','joint','equal','k','tie_order']
    assert not any(isinstance(n,ast.Name) and n.id in ('open','eval','exec','__import__') for n in ast.walk(tree))
    manifest=json.loads((ROOT/p['inputs']).read_text());e=next(e for e in manifest['inputs'] if e['seed']==a.seed)
    assert sha(ROOT/p['inputs'])==p['input_manifest_sha256']==r['input_manifest_sha256']
    assert sha(ROOT/e['completion_path'])==r['source_completion_sha256']==e['completion_sha256']
    full=np.load(ROOT/e['completion_path']);observed={}
    for entry in r['observations']:
        assert sha(ROOT/entry['path'])==entry['sha256'];z=np.load(ROOT/entry['path'])
        assert set(z.files)=={'values','given','source_row'} and np.array_equal(z['source_row'],full['source_row'])
        assert np.all(z['given'].sum(1)==25) and np.all(z['values'][~z['given']]==-1)
        assert np.array_equal(z['values'][z['given']],full['solutions'][z['given']])
        observed[entry['arm']]={k:z[k] for k in ['values','given']}
    assert sha(d/'acquisition_events.json')==r['acquisition_events_sha256']
    events=json.loads((d/'acquisition_events.json').read_text());assert len(events)==3000
    states={arm:(np.zeros((81,81),np.int32),np.zeros((81,81),np.int32)) for arm in p['arms']}
    rng=np.random.default_rng(20261125+a.seed);checks=0;pair=np.triu_indices(81,1)
    for row in range(1000):
        order=rng.permutation(81)
        for ai,arm in enumerate(p['arms']):
            joint,equal=states[arm];event=events[row*3+ai]
            assert event['row']==row and event['arm']==arm and event['events']==['finalized','revealed']
            assert event['past_counts_sha256']==hashlib.sha256(joint.tobytes()+equal.tobytes()).hexdigest()
            expected=reference_mask(arm,joint,equal,order);mask=observed[arm]['given'][row]
            assert np.array_equal(expected,mask)
            assert event['mask_sha256']==hashlib.sha256(mask.tobytes()).hexdigest()
            # Update from the archived queried values only, after checking the current mask.
            joint+=(mask[:,None]&mask[None,:]).astype(np.int32)
            values=observed[arm]['values'][row]
            for digit in range(9):
                cells=np.flatnonzero(mask&(values==digit));equal[np.ix_(cells,cells)]+=1
            checks+=1
            n=row+1
            if n in p['sizes']:
                entry=next(e for e in r['models'] if e['arm']==arm and e['n']==n)
                assert sha(ROOT/entry['path'])==entry['sha256'];m=np.load(ROOT/entry['path'])
                assert np.array_equal(m['equal_count'],equal[pair]) and np.array_equal(m['jointly_given_count'],joint[pair])
                assert np.array_equal(m['support'],equal[pair]==0)
        if row%500==0:save(d/'replay_progress.json',{'utc':now(),'rows_replayed':row+1,'elapsed_s':time.monotonic()-start})
    hidden_checks=0
    for arm,z in observed.items():
        for n in p['sizes']:
            entry=next(e for e in r['models'] if e['arm']==arm and e['n']==n);m=np.load(ROOT/entry['path'])
            values=z['values'][:n].copy();given=z['given'][:n]
            for replacement in ('sentinel','shuffled_hidden'):
                if replacement=='sentinel':values[~given]=-99
                else:values[~given]=np.random.default_rng(20261226+a.seed+n).permutation(full['solutions'][:n][~given])
                support,equal,both=estimate(values,given)
                assert np.array_equal(support,m['support']) and np.array_equal(equal,m['equal_count'])
                assert np.array_equal(both,m['jointly_given_count']);hidden_checks+=1
    rec={'status':'validated','utc':now(),'seed':a.seed,'protocol_sha256':sha(PROTOCOL),
        'wrapper_sha256':sha(__file__),'acquisition_receipt_sha256':sha(d/'receipt.json'),
        'independent_masks_replayed':checks,'prefix_supports_verified':12,'hidden_value_recount_checks':hidden_checks,
        'source_digit_and_row_identity_verified':True,'current_digit_policy_interface_excluded':True,
        'elapsed_s':time.monotonic()-start,'numpy':np.__version__}
    save(d/'acquisition_audit.json',rec);print(json.dumps(rec),flush=True)


if __name__=='__main__':main()
