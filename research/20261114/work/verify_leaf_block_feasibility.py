"""Public-only H2ag full-row primal check with independent source and move audit."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from verify_training_feasibility import energy,flip,LD
from audit_leaf_selection import verify as source_verify,components


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(directory,training):
    started=time.perf_counter();p=json.loads((directory/'protocol.json').read_text());solver=json.loads((directory/'solver-summary.json').read_text());moves=json.loads((directory/'move-selection.json').read_text());source=directory/'source';assert sha(source/'protocol.json')==p['parent_leaf_protocol_sha256'] and sha(directory/'move-selection.json')==p['selection_sha256'];assert solver['status']=='complete' and solver['protocol_sha256']==sha(directory/'protocol.json');source_audit=source_verify(source,training);assert source_audit['status']=='passed';old_selection=json.loads((source/'leaf-selection.json').read_text())
    with np.load(training/'training-arrays.npz') as z:tokens=z['tokens'];labels=z['labels']
    rows=[]
    for seed in range(3):
        record=solver['records'][seed];selected=moves['records'][seed];original=old_selection['records'][seed];assert selected['mapping']==original['mapping'] and selected['eligible_leaves']==original['eligible_leaves'];assert record['status']=='verified_feasible_primal'
        with np.load(source/f'initial-seed{seed}.npz') as z:mu=z['mu_offsets'];ao=z['aoffs']
        with np.load(training/f'train-readout-seed{seed}.npz') as z:attention=z['couplings']
        cp=directory/f'seed{seed}/candidate-theta.npy';assert sha(cp)==record['candidate_sha256'];theta=np.load(cp).astype(LD)
        op=directory/f'old-negatives-seed{seed}.npz';assert sha(op)==p['previous_witnesses'][str(seed)]['negative_archive_sha256']
        with np.load(op) as z:ix=z['indices'];boards=z['boards'];ham=z['hamming']
        with np.load(source/f'negatives-seed{seed}.npz') as z:current_ix=z['indices'];current_boards=z['boards']
        expected=[]
        for item in original['wrong_boards_without_leaf']:
            j=item['negative_row'];i=item['training_index'];b=current_boards[j];assert int(current_ix[j])==i
            for cells in components((tokens[i]==1)&b&~labels[i]):expected.append((j,i,tuple(sorted(cells))))
        assert sorted(expected)==sorted((r['negative_row'],r['training_index'],tuple(sorted(map(tuple,r['cells'])))) for r in selected['blocks'])
        minima=np.full(6,np.inf,LD);counts=np.zeros(6,np.int64)
        for lo in range(0,1000,64):
            t=tokens[lo:lo+64];y=labels[lo:lo+64];a=attention[lo:lo+64].astype(LD);free=t==1;canonical=energy(t,y,a,theta,mu,ao);on=(free&y).sum((1,2));off=(free&~y).sum((1,2));values=[flip(t,y,a,theta,mu,ao)[free],(energy(t,np.zeros_like(y),a,theta,mu,ao)-canonical)/on,(energy(t,np.ones_like(y),a,theta,mu,ao)-canonical)/off]
            for k,v in enumerate(values):minima[k]=min(minima[k],v.min());counts[k]+=v.size
        for lo in range(0,len(ix),64):
            indices=ix[lo:lo+64];t=tokens[indices];a=attention[indices].astype(LD);v=(energy(t,boards[lo:lo+64],a,theta,mu,ao)-energy(t,labels[indices],a,theta,mu,ao))/ham[lo:lo+64];minima[3]=min(minima[3],v.min());counts[3]+=len(v)
        for item in selected['mapping']:
            i=item['training_index'];b=current_boards[item['negative_row']];v=-flip(tokens[i:i+1],b[None],attention[i:i+1].astype(LD),theta,mu,ao)[0].flat[item['flat']];minima[4]=min(minima[4],v);counts[4]+=1
        duplicates=[]
        for item in selected['blocks']:
            i=item['training_index'];b=current_boards[item['negative_row']];z=b.copy()
            for r,c in item['cells']:assert z[r,c] and not labels[i,r,c] and tokens[i,r,c]==1;z[r,c]=False
            assert np.array_equal(z,labels[i]) and item['size']==len(item['cells']);matches=[k for k,idx in enumerate(ix) if int(idx)==i and np.array_equal(boards[k],b) and int(ham[k])==item['size']];assert matches;duplicates.append({'block_row':item['block_row'],'exact_duplicate_prior_negative_rows':matches})
            a=attention[i:i+1].astype(LD);t=tokens[i:i+1];v=(energy(t,b[None],a,theta,mu,ao)-energy(t,z[None],a,theta,mu,ao))[0]/item['size'];minima[5]=min(minima[5],v);counts[5]+=1
        norm=abs(theta).max();m=LD(256*900+1000);u=LD(2)**-53;gamma=m*u/(1-m*u);allowance=4*gamma*(10000*max(LD(1),norm)+1);assert np.isfinite(minima).all() and np.all(minima-allowance>1);assert counts.tolist()==record['constraints_by_kind']
        rows.append({'seed':seed,'constraints_by_kind':counts.tolist(),'minimum_margins':list(map(float,minima)),'conservative_absolute_allowance':float(allowance),'certified_margin_lower':float(minima.min()-allowance),'candidate_linf':float(norm),'candidate_sha256':sha(cp),'component_duplicate_audit':duplicates})
    return {'status':'passed','hypothesis':'H2ag','decision':'supported_for_stated_source_constraints','rows':rows,'constraints_verified':sum(sum(r['constraints_by_kind']) for r in rows),'leaf_rows':sum(r['constraints_by_kind'][4] for r in rows),'duplicate_component_rows':sum(r['constraints_by_kind'][5] for r in rows),'source_audit':source_audit,'new_source_trajectories':0,'new_independent_test_inputs':0,'seconds':time.perf_counter()-started,'protocol_sha256':sha(directory/'protocol.json'),'verifier_sha256':sha(Path(__file__)),'scope':'All old rows, all eligible source leaves and every leafless component retained. Independent source identities/masks/graphs; direct extended-precision primal inequalities. Exact component duplication checked by identical input, board, removed cells and normalization. No minimum-norm optimality or decoder-generalization claim.'}


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('directory',type=Path);a.add_argument('training',type=Path);v=a.parse_args();print(json.dumps(verify(v.directory,v.training),indent=2))
