"""Independent retained-source graph, scale, leaf and energy diagnosis for H2af."""
from collections import deque
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from verify_training_feasibility import energy,flip,LD
from verify_hard_negative_feasibility import valid_shortest


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def components(mask):
    unseen=set(map(tuple,np.argwhere(mask)));parts=[]
    while unseen:
        first=unseen.pop();part=[first];queue=[first]
        while queue:
            r,c=queue.pop()
            for q in [(r-1,c),(r,c+1),(r+1,c),(r,c-1)]:
                if q in unseen:unseen.remove(q);part.append(q);queue.append(q)
        parts.append(part)
    return parts


def exact_leaf(t,b,a,mu,ao,flat):
    p=np.where(t==1,b,t>=2);h,w=t.shape;r,c=divmod(flat,w);assert p[r,c] and t[r,c]==1;x=[Q(0)]*23;x[16]=Q(1);x[17]=Q(-1)
    for k,(dr,dc) in enumerate(mu):
        dr,dc=int(dr),int(dc)
        if 0<=r+dr<h and 0<=c+dc<w:v=int(p[r+dr,c+dc]);x[4*k+2+v]+=1;x[4*k+v]-=1
        if 0<=r-dr<h and 0<=c-dc<w:v=int(p[r-dr,c-dc]);x[4*k+2*v+1]+=1;x[4*k+2*v]-=1
    for k,(dr,dc) in enumerate(ao):
        dr,dc=int(dr),int(dc)
        if dr==dc==0:x[18+k]=-Q.from_float(float(a[k,r,c]));continue
        if 0<=r+dr<h and 0<=c+dc<w and p[r+dr,c+dc]:x[18+k]-=Q.from_float(float(a[k,r,c]))
        if 0<=r-dr<h and 0<=c-dc<w and p[r-dr,c-dc]:x[18+k]-=Q.from_float(float(a[k,r-dr,c-dc]))
    return x


def verify(directory,training):
    started=time.perf_counter();p=json.loads((directory/'protocol.json').read_text());selection=json.loads((directory/'leaf-selection.json').read_text());mining=json.loads((directory/'mining-audit.json').read_text());assert selection['protocol_sha256']==sha(directory/'protocol.json') and sha(training/'training-arrays.npz')==p['training_manifest']['arrays_sha256']
    with np.load(training/'training-arrays.npz') as z:tokens=z['tokens'];labels=z['labels']
    assert all(valid_shortest(t,y) for t,y in zip(tokens,labels));rows=[];leafless=[]
    for seed in range(3):
        record=selection['records'][seed];receipt=json.loads((directory/f'mined-seed{seed}.json').read_text());assert sha(directory/f'mined-seed{seed}.npz')==receipt['outputs_sha256']
        with np.load(directory/f'mined-seed{seed}.npz') as z:pred=z['prediction'];stopped=z['stopped'];finite=z['finite'];saved_margin=z['minimum_flip_margin']
        actual=np.flatnonzero([not valid_shortest(t,b) for t,b in zip(tokens,pred)])
        with np.load(directory/f'negatives-seed{seed}.npz') as z:ix=z['indices'];boards=z['boards'];hamming=z['hamming']
        assert np.array_equal(ix,actual) and np.array_equal(boards,pred[ix]);assert len(ix)==record['wrong_boards']==mining['records'][seed]['negative_count']
        initial=directory/f'initial-seed{seed}.npz';assert sha(initial)==p['initial_models'][seed]['sha256']
        with np.load(initial) as z:original=np.concatenate([z['T'].ravel(),z['b'],z['kappa']]);mu=z['mu_offsets'];ao=z['aoffs']
        candidate=directory/f'previous-theta-seed{seed}.npy';assert sha(candidate)==p['previous_witnesses'][str(seed)]['sha256'];theta=np.load(candidate);cast=np.asarray(theta*(float(abs(original).max())/float(abs(theta).max())),np.float32);cp=directory/f'current-cast-seed{seed}.npy';assert sha(cp)==p['coefficient_arrays'][str(seed)]['sha256'] and np.array_equal(cast,np.load(cp));theta=cast.astype(LD)
        ap=training/f'train-readout-seed{seed}.npz';assert sha(ap)==p['readouts'][str(seed)]['readout_sha256']
        with np.load(ap) as z:attention=z['couplings']
        maxerror=0.;minimum_label=np.inf
        for lo in range(0,1000,64):
            t=tokens[lo:lo+64];b=pred[lo:lo+64];y=labels[lo:lo+64];a=attention[lo:lo+64].astype(LD);d=flip(t,b,a,theta,mu,ao);m=np.where(t==1,d,np.inf).reshape(len(t),-1).min(1);maxerror=max(maxerror,float(abs(m-saved_margin[lo:lo+64]).max()));assert (m>=-1e-8).all();minimum_label=min(minimum_label,float(flip(t,y,a,theta,mu,ao)[t==1].min()))
        assert maxerror<1e-10 and stopped.all() and finite.all() and minimum_label>0
        selected=[];without=[];exact_costs=[];known=set();structures=[]
        for j,(i,b) in enumerate(zip(ix,boards)):
            t=tokens[i];y=labels[i];degree=np.zeros_like(t,np.int8)
            for r,c in np.argwhere(b):
                for rr,cc in [(r-1,c),(r,c+1),(r+1,c),(r,c-1)]:
                    if 0<=rr<30 and 0<=cc<30:degree[rr,cc]+=1
            positions=np.flatnonzero((t==1)&b&~y&(degree==1));ncomp=len(components(b));edge=int((b[1:]&b[:-1]).sum()+(b[:,1:]&b[:,:-1]).sum());structure={'training_index':int(i),'negative_row':j,'components':ncomp,'cycles_rank':edge-int(b.sum())+ncomp,'branch_nodes':int((b&(degree>2)).sum()),'missed_label_cells':int((y&~b).sum()),'extra_cells':int((b&~y).sum()),'eligible_leaves':len(positions)};structures.append(structure)
            if not len(positions):
                without.append({'negative_row':j,'training_index':int(i)});base=energy(t[None],b[None],attention[i:i+1].astype(LD),theta,mu,ao)[0];parts=components(b&~y&(t==1));diagnostics=[]
                for cells in parts:
                    z=b.copy()
                    for r,c in cells:z[r,c]=False
                    change=energy(t[None],z[None],attention[i:i+1].astype(LD),theta,mu,ao)[0]-base;diagnostics.append({'size':len(cells),'cells':[list(map(int,q)) for q in cells],'removal_energy_change_cast':float(change),'removal_cost_per_cell_cast':float(change/len(cells))})
                leafless.append({'seed':seed,**structure,'false_positive_four_neighbor_components':diagnostics})
            for f in positions:
                item=record['mapping'][len(selected)];assert item['negative_row']==j and item['training_index']==int(i) and item['flat']==int(f) and item['selected_neighbor_count']==1
                vector=exact_leaf(t,b,attention[i],mu,ao,int(f));cost=-sum(x*Q.from_float(float(v)) for x,v in zip(vector,theta));assert cost==Q(int(item['previous_exact_flip_numerator']),int(item['previous_exact_flip_denominator']));digest=hashlib.sha256(json.dumps([[v.numerator,v.denominator] for v in vector],separators=(',',':')).encode()).hexdigest();assert digest==item['exact_row_sha256'] and (digest not in known)==item['new_exact_row_within_round'];known.add(digest);selected.append(item);exact_costs.append(cost)
        assert without==record['wrong_boards_without_leaf'] and len(selected)==record['eligible_leaves'] and len(known)==record['unique_new_leaf_rows']
        rows.append({'seed':seed,'wrong_source_outputs':len(ix),'eligible_leaves':len(selected),'unique_exact_leaf_rows':len(known),'leafless_boards':without,'zero_cost_leaves_exact':sum(x==0 for x in exact_costs),'negative_cost_leaves_exact':sum(x<0 for x in exact_costs),'positive_cost_leaves_exact':sum(x>0 for x in exact_costs),'minimum_leaf_flip_cost':float(min(exact_costs)),'maximum_leaf_flip_cost':float(max(exact_costs)),'all_source_outputs_certified':1000,'minimum_canonical_label_flip':minimum_label,'maximum_direct_vs_saved_margin_error':maxerror,'source_structures':structures})
    return {'status':'passed','decision':'uninformative_no_leaf_gate','rows':rows,'leafless_diagnostics':leafless,'source_outputs_independently_classified':3000,'source_states_direct_energy_checked':3000,'canonical_training_labels_checked':1000,'new_independent_test_inputs':0,'seconds':time.perf_counter()-started,'protocol_sha256':sha(directory/'protocol.json'),'verifier_sha256':sha(Path(__file__)),'scope':'Independent graph classification, exact scaling/cast, all saved-state certificates, every leaf mask and exact leaf-cost coefficient. Leafless component removals are post-outcome source diagnostics, no fit or new trajectories. Cached/reference full source replays are separate.'}


if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('directory',type=Path);a.add_argument('training',type=Path);v=a.parse_args();print(json.dumps(verify(v.directory,v.training),indent=2))
