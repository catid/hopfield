"""Prospective fresh-bank harness using the unchanged policy; masks precede reveal."""
import argparse
import hashlib
import json
import time
import numpy as np
from recovery_common import load_protocol
from run import ROOT,HERE,save,sha,now
from exponential_policy import choose_mask as exponential_choose_mask,learn_queried
from online_policy import choose_mask as original_choose_mask

def choose_mask(arm,joint,equal,k,tie_order):
    return (original_choose_mask if arm=='uniform' else exponential_choose_mask)(arm,joint,equal,k,tie_order)

OUT=ROOT/'artifacts/research/20261114/ripper/recovery'
PROTOCOL=HERE/'protocol-recovery.json'


class QueryOracle:
    def __init__(self,row):
        self.__row=row.copy();self.__mask=None;self.events=[]

    def finalize(self,mask):
        assert self.__mask is None and mask.shape==(81,) and mask.sum()==25
        self.__mask=mask.copy();self.__mask.setflags(write=False);self.events.append('finalized')

    def reveal(self):
        if self.__mask is None: raise RuntimeError('Mask must be finalized before reveal')
        assert self.events==['finalized'];self.events.append('revealed')
        return self.__row[self.__mask].copy()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);a=ap.parse_args()
    assert np.__version__=='2.3.4'
    p=load_protocol();assert sha(HERE/'exponential_policy.py')==p['frozen_policy_sha256'];mp=ROOT/p['inputs'];assert sha(mp)==p['input_manifest_sha256']
    manifest=json.loads(mp.read_text());e=next(e for e in manifest['inputs'] if e['seed']==a.seed)
    source=ROOT/e['completion_path'];assert sha(source)==e['completion_sha256'];z=np.load(source)
    solutions=z['solutions'];source_rows=z['source_row'];assert len(solutions)==1000
    d=OUT/f's{a.seed}';d.mkdir(parents=True,exist_ok=False)
    states={arm:{'joint':np.zeros((81,81),np.int32),'equal':np.zeros((81,81),np.int32),
        'given':np.zeros((1000,81),bool),'values':np.full((1000,81),-1,np.int8)} for arm in p['arms']}
    rng=np.random.default_rng(20261125+a.seed);shadow_rng=np.random.default_rng(20261225+a.seed)
    pair=np.triu_indices(81,1);models=[];gates=0;premature=0;events=[];start=time.monotonic();selection_s=0.;reveal_s=0.
    for row in range(1000):
        order=rng.permutation(81);shadow_row=solutions[row,shadow_rng.permutation(81)]
        for arm,state in states.items():
            joint,equal=state['joint'],state['equal'];before=hashlib.sha256(joint.tobytes()+equal.tobytes()).hexdigest()
            joint.setflags(write=False);equal.setflags(write=False)
            oracle=QueryOracle(solutions[row]);shadow=QueryOracle(shadow_row)
            try:oracle.reveal()
            except RuntimeError:premature+=1
            else:raise AssertionError('Premature reveal permitted')
            t=time.monotonic();mask=choose_mask(arm,joint,equal,25,order)
            oracle.finalize(mask)
            shadow_mask=choose_mask(arm,joint,equal,25,order);shadow.finalize(shadow_mask)
            assert np.array_equal(mask,shadow_mask)
            assert hashlib.sha256(joint.tobytes()+equal.tobytes()).hexdigest()==before
            selection_s+=time.monotonic()-t;gates+=1
            # Only now does the actual policy receive the 25 current labels.
            t=time.monotonic();queried=oracle.reveal();assert oracle.events==['finalized','revealed']
            assert shadow.events==['finalized']
            state['given'][row]=mask;state['values'][row,mask]=queried
            joint.setflags(write=True);equal.setflags(write=True)
            learn_queried(joint,equal,mask,queried);reveal_s+=time.monotonic()-t
            events.append({'row':row,'arm':arm,'past_counts_sha256':before,
                'mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest(),'events':oracle.events})
            n=row+1
            if n in p['sizes']:
                path=d/f'{arm}-n{n}.npz'
                np.savez_compressed(path,support=equal[pair]==0,equal_count=equal[pair],jointly_given_count=joint[pair])
                models.append({'seed':a.seed,'n':n,'arm':arm,'path':str(path.relative_to(ROOT)),
                    'sha256':sha(path),'support_bytes_sha256':hashlib.sha256((equal[pair]==0).tobytes()).hexdigest()})
        if row+1 in p['sizes']:
            save(d/f'models-n{row+1}.json',{'status':'prepared_without_geometry','utc':now(),
                'protocol_sha256':sha(PROTOCOL),'wrapper_sha256':sha(__file__),
                'policy_sha256':sha(HERE/'exponential_policy.py'),'models':[m for m in models if m['n']==row+1]})
        if row%100==0 or row==999:
            save(d/'progress.json',{'utc':now(),'rows':row+1,'total':1000,'selection_s':selection_s,
                'reveal_s':reveal_s,'elapsed_s':time.monotonic()-start})
    observed=[]
    for arm,state in states.items():
        path=d/f'{arm}-observations.npz'
        np.savez_compressed(path,values=state['values'],given=state['given'],source_row=source_rows)
        observed.append({'arm':arm,'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
    save(d/'acquisition_events.json',events)
    save(d/'receipt.json',{'status':'complete','utc':now(),'seed':a.seed,'protocol_sha256':sha(PROTOCOL),
        'wrapper_sha256':sha(__file__),'policy_sha256':sha(HERE/'exponential_policy.py'),
        'input_manifest_sha256':sha(mp),'source_completion_sha256':sha(source),'models':models,'observations':observed,
        'current_row_permutation_invariance_checks':gates,'premature_reveal_rejections':premature,
        'acquisition_events_sha256':sha(d/'acquisition_events.json'),'selection_s':selection_s,'reveal_s':reveal_s,
        'elapsed_s':time.monotonic()-start,'numpy':np.__version__,
        'cost':'Exactly25 queried digits per board per policy;7500 atN300,11250 atN450,15000 atN600,25000 atN1000. Stored completions are the simulated oracle, not policy inputs.'})
    print(json.dumps({'seed':a.seed,'rows':1000,'gates':gates,'elapsed_s':time.monotonic()-start}),flush=True)


if __name__=='__main__':main()
