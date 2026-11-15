"""Public-only verification of source negatives and every H2ab inequality."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from verify_training_feasibility import energy,flip,LD


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_shortest(t,b):
    h,w=t.shape;start=tuple(np.argwhere(t==2)[0]);goal=tuple(np.argwhere(t==3)[0]);distance={start:0};queue=deque([start])
    while queue:
        r,c=queue.popleft()
        for rr,cc in [(r-1,c),(r,c+1),(r+1,c),(r,c-1)]:
            if 0<=rr<h and 0<=cc<w and t[rr,cc]!=0 and (rr,cc) not in distance:distance[(rr,cc)]=distance[(r,c)]+1;queue.append((rr,cc))
    if goal not in distance or not b[start] or not b[goal] or np.any(b&(t==0)):return False
    degree=np.zeros_like(t,dtype=np.int8);degree[1:]+=b[:-1];degree[:-1]+=b[1:];degree[:,1:]+=b[:,:-1];degree[:,:-1]+=b[:,1:]
    if degree[start]!=1 or degree[goal]!=1 or np.any(degree[b&(t==1)]!=2):return False
    seen={start};queue=deque([start])
    while queue:
        r,c=queue.popleft()
        for rr,cc in [(r-1,c),(r,c+1),(r+1,c),(r,c-1)]:
            if 0<=rr<h and 0<=cc<w and b[rr,cc] and (rr,cc) not in seen:seen.add((rr,cc));queue.append((rr,cc))
    return len(seen)==int(b.sum())==distance[goal]+1


def verify(directory,training):
    started=time.perf_counter();p=json.loads((directory/'protocol.json').read_text());solver=json.loads((directory/'solver-summary.json').read_text());mining=json.loads((directory/'mining-audit.json').read_text());assert solver['status']=='complete' and solver['protocol_sha256']==sha(directory/'protocol.json') and mining['protocol_sha256']==sha(directory/'protocol.json')
    assert sha(training/'training-arrays.npz')==p['training_manifest']['arrays_sha256']
    with np.load(training/'training-arrays.npz',allow_pickle=False) as z:tokens=z['tokens'];labels=z['labels']
    assert tokens.shape==labels.shape==(1000,30,30) and labels.dtype==bool
    assert all(valid_shortest(t,b) for t,b in zip(tokens,labels));rows=[]
    for seed in range(3):
        record=solver['records'][seed];assert record['seed']==seed
        receipt=json.loads((directory/f'mined-seed{seed}.json').read_text());assert receipt['protocol_sha256']==sha(directory/'protocol.json')
        path=directory/f'mined-seed{seed}.npz';assert sha(path)==receipt['outputs_sha256']
        with np.load(path,allow_pickle=False) as z:pred=z['prediction'];finite=z['finite'];stopped=z['stopped'];mined_margin=z['minimum_flip_margin']
        assert pred.shape==(1000,30,30) and pred.dtype==bool and finite.all()
        actual=np.flatnonzero([not valid_shortest(t,b) for t,b in zip(tokens,pred)])
        path=directory/f'negatives-seed{seed}.npz';mine=mining['records'][seed];assert sha(path)==mine['negative_archive_sha256']
        with np.load(path,allow_pickle=False) as z:indices=z['indices'];boards=z['boards'];hamming=z['hamming']
        assert np.array_equal(indices,actual) and np.array_equal(boards,pred[indices]);assert len(indices)>0
        assert np.array_equal(hamming,((boards!=labels[indices])&(tokens[indices]==1)).sum((1,2))) and (hamming>0).all()
        assert len(indices)==mine['negative_count']==record['mined_negatives']==len(mine['mapping'])
        for j,row in enumerate(mine['mapping']):
            assert row['negative_row']==j and row['training_index']==int(indices[j]) and row['free_hamming']==int(hamming[j]) and row['board_sha256']==hashlib.sha256(boards[j].tobytes()).hexdigest()
            assert row['certified']==bool(stopped[indices[j]] and mined_margin[indices[j]]>=-1e-8)
        candidate=directory/f'seed{seed}/candidate-theta.npy';assert sha(candidate)==record['candidate_sha256'];theta=np.load(candidate,allow_pickle=False).astype(LD);assert theta.shape==(23,) and np.isfinite(theta).all()
        cp=directory/f'initial-seed{seed}.npz';assert sha(cp)==p['initial_models'][seed]['sha256']
        with np.load(cp,allow_pickle=False) as z:muoffs=z['mu_offsets'];aoffs=z['aoffs']
        ap=training/f'train-readout-seed{seed}.npz';assert sha(ap)==p['readouts'][str(seed)]['readout_sha256']
        with np.load(ap,allow_pickle=False) as z:attention=z['couplings']
        minima=np.full(4,np.inf,LD);counts=np.zeros(4,np.int64)
        for lo in range(0,1000,64):
            t=tokens[lo:lo+64];y=labels[lo:lo+64];a=attention[lo:lo+64].astype(LD);free=t==1;on=(free&y).sum((1,2));off=(free&~y).sum((1,2));canonical=energy(t,y,a,theta,muoffs,aoffs)
            values=[flip(t,y,a,theta,muoffs,aoffs)[free],(energy(t,np.zeros_like(y),a,theta,muoffs,aoffs)-canonical)/on,(energy(t,np.ones_like(y),a,theta,muoffs,aoffs)-canonical)/off]
            for k,value in enumerate(values):assert np.isfinite(value).all();minima[k]=min(minima[k],value.min());counts[k]+=value.size
        for lo in range(0,len(indices),64):
            ix=indices[lo:lo+64];t=tokens[ix];y=labels[ix];a=attention[ix].astype(LD);z=boards[lo:lo+64];value=(energy(t,z,a,theta,muoffs,aoffs)-energy(t,y,a,theta,muoffs,aoffs))/hamming[lo:lo+64];assert np.isfinite(value).all();minima[3]=min(minima[3],value.min());counts[3]+=len(ix)
        norm=abs(theta).max();m=LD(256*900+1000);u=LD(2)**-53;gamma=m*u/(1-m*u);allowance=4*gamma*(10000*max(LD(1),norm)+1);assert np.all(minima-allowance>1)
        assert counts.tolist()==[564329,1000,1000,len(indices)]
        rows.append({'seed':seed,'negative_count':len(indices),'all_source_outputs_independently_classified':1000,'constraints_by_kind':counts.tolist(),'minimum_normalized_margins':list(map(float,minima)),'conservative_float64_allowance':float(allowance),'certified_margin_lower':float(minima.min()-allowance),'parameter_linf':float(norm),'candidate_sha256':sha(candidate)})
    return {'status':'passed','hypothesis':'H2ab','decision':'supported_for_stated_source_constraints','rows':rows,'source_outputs_classified':3000,'canonical_training_labels_checked':1000,'retained_negative_count':sum(r['negative_count'] for r in rows),'constraints_verified':sum(sum(r['constraints_by_kind']) for r in rows),'new_independent_test_inputs':0,'seconds':time.perf_counter()-started,'verifier_sha256':sha(Path(__file__)),'direct_energy_helper_sha256':sha(Path(__file__).with_name('verify_training_feasibility.py')),'protocol_sha256':sha(directory/'protocol.json'),'scope':'Alloriginaltrainingarrays/fixedreadouts only. IndependentsourceBFS/degree/connectivity classification andallnormalizedinequalities via directextendedprecisionbinaryenergy/flip arithmetic. No Torch, SciPy, numericalsolver orlinearfeaturebuilder. Feasiblewitnesses do not certify optimum norm, global labeloptimality, source decoder success afterrefitting, or testgeneralization.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('training',type=Path);args=parser.parse_args();print(json.dumps(verify(args.directory,args.training),indent=2))
