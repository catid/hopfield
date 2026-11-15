"""Public-only direct-rational Hölder certificate and full feasible-primal audit."""
import argparse
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import verify_training_feasibility as V


def dec(d):return Q(int(d['n']),int(d['d']))


def binary_vector(token,board,attention,muoffs,aoffs):
    """Exact direct sum of state-pair indicators and individual active links."""
    p=np.where(token==1,board,token>=2).astype(np.int8);height,width=token.shape;vector=[Q(0)]*23
    for k,(dr,dc) in enumerate(muoffs):
        dr,dc=int(dr),int(dc)
        for r in range(height):
            for c in range(width):
                rr,cc=r+dr,c+dc
                if 0<=rr<height and 0<=cc<width:vector[4*k+2*int(p[r,c])+int(p[rr,cc])]+=1
    for r in range(height):
        for c in range(width):
            if token[r,c]==1:vector[16+int(p[r,c])]-=1
    for k,(dr,dc) in enumerate(aoffs):
        dr,dc=int(dr),int(dc)
        for r in range(height):
            for c in range(width):
                rr,cc=r+dr,c+dc
                if 0<=rr<height and 0<=cc<width and p[r,c] and p[rr,cc]:vector[18+k]-=Q.from_float(float(attention[k,r,c]))
    return list(map(Q,vector))


def verify(directory,training):
    started=time.perf_counter();protocol=json.loads((directory/'protocol.json').read_text());summary=json.loads((directory/'solver-summary.json').read_text());assert summary['status']=='complete' and summary['protocol_sha256']==V.sha(directory/'protocol.json');assert V.sha(training/'training-arrays.npz')==protocol['training_manifest']['arrays_sha256']
    with np.load(training/'training-arrays.npz',allow_pickle=False) as x:tokens=x['tokens'];labels=x['labels']
    records=[]
    for result in summary['records']:
        seed=result['seed'];folder=directory/f'seed{seed}';certpath=folder/'norm-certificate.json';assert V.sha(certpath)==result['certificate_sha256'];cert=json.loads(certpath.read_text());assert V.sha(folder/'candidate-theta.npy')==result['candidate_sha256'];theta=np.load(folder/'candidate-theta.npy',allow_pickle=False).astype(V.LD);assert theta.shape==(23,)
        cp=directory/f'initial-seed{seed}.npz';assert V.sha(cp)==protocol['initial_models'][seed]['sha256']
        with np.load(cp,allow_pickle=False) as z:muoffs=z['mu_offsets'];aoffs=z['aoffs'];initial=np.concatenate([z['T'].ravel(),z['b'],z['kappa']])
        ap=training/f'train-readout-seed{seed}.npz';assert V.sha(ap)==protocol['readouts'][str(seed)]['readout_sha256']
        with np.load(ap,allow_pickle=False) as z:attention=z['couplings']
        constants={}
        for item in cert['constants']:
            path=folder/item['path'];assert V.sha(path)==item['sha256']
            with np.load(path,allow_pickle=False) as z:state={k:z[k] for k in z.files}
            i=item['training_index'];assert np.array_equal(state['tokens'],tokens[i]) and np.array_equal(state['label'],labels[i]) and np.array_equal(state['attention'],attention[i]);assert np.array_equal(state['mu_offsets'],muoffs) and np.array_equal(state['aoffs'],aoffs);constants[i]=state
        exact=[]
        for meta,stored in zip(cert['constraints'],cert['rows']):
            z=constants[meta['training_index']];token=z['tokens'];label=z['label'];a=z['attention'];canonical=binary_vector(token,label,a,muoffs,aoffs);kind=meta['kind']
            if kind==0:
                negative=label.copy();flat=meta['flat_index'];assert token.ravel()[flat]==1;negative.ravel()[flat]=not negative.ravel()[flat];normalizer=1
            elif kind==1:negative=np.zeros_like(label);normalizer=int(np.count_nonzero((token==1)&label))
            else:assert kind==2;negative=np.ones_like(label);normalizer=int(np.count_nonzero((token==1)&~label))
            other=binary_vector(token,negative,a,muoffs,aoffs);row=[(b-a)/normalizer for a,b in zip(canonical,other)];assert row==list(map(dec,stored));exact.append(row)
        weights=list(map(dec,cert['weights']));assert len(weights)==len(exact) and all(x>=0 for x in weights);numerator=sum(weights,Q(0));gradient=[sum((w*row[j] for w,row in zip(weights,exact)),Q(0)) for j in range(23)];denominator=sum(map(abs,gradient),Q(0));assert numerator>0 and denominator>0;lower=numerator/denominator
        assert numerator==dec(cert['numerator']) and denominator==dec(cert['denominator']) and gradient==list(map(dec,cert['gradient'])) and lower==dec(cert['lower_bound'])
        minima=np.full(3,np.inf,V.LD);counts=np.zeros(3,np.int64)
        for lo in range(0,1000,64):
            t=tokens[lo:lo+64];y=labels[lo:lo+64];a=attention[lo:lo+64].astype(V.LD);free=t==1;on=(free&y).sum((1,2));off=(free&~y).sum((1,2));canonical=V.energy(t,y,a,theta,muoffs,aoffs);values=[V.flip(t,y,a,theta,muoffs,aoffs)[free],(V.energy(t,np.zeros_like(y),a,theta,muoffs,aoffs)-canonical)/on,(V.energy(t,np.ones_like(y),a,theta,muoffs,aoffs)-canonical)/off]
            for k,value in enumerate(values):minima[k]=min(minima[k],value.min());counts[k]+=value.size
        norm=np.abs(theta).max();m=V.LD(256*900+1000);u=V.LD(2)**-53;gamma=m*u/(1-m*u);allowance=4*gamma*(10000*max(V.LD(1),norm)+1);assert np.all(minima-allowance>1) and counts.tolist()==[564329,1000,1000];upper=Q.from_float(float(norm));assert upper==dec(cert['upper_bound']) and lower<=upper;gap=(upper-lower)/upper
        parts={str(k):sum((weight for weight,meta in zip(weights,cert['constraints']) if meta['kind']==k),Q(0))/denominator for k in range(3)};assert sum(parts.values(),Q(0))==lower
        tradeoff={'exact_coefficient_by_constraint_kind':{k:{'n':str(v.numerator),'d':str(v.denominator)} for k,v in parts.items()},'coefficient_by_constraint_kind':{k:float(v) for k,v in parts.items()},'formula':'||theta||inf >= c0*local_flip_margin+c1*endpoint_normalized_margin+c2*dense_normalized_margin for these sameconstraintrows; exactweightedinequality, no newsolve.','nonnegative_local_and_unit_endpoint_lower':float(parts['1']),'nonnegative_local_and_unit_endpoint_lower_over_original':float(parts['1'])/float(abs(initial).max())}
        records.append({'seed':seed,'status':'primal_and_exact_dual_bound_verified','exact_lower_bound':cert['lower_bound'],'exact_upper_bound':cert['upper_bound'],'lower_bound':float(lower),'upper_bound':float(upper),'relative_gap':float(gap),'relative_gap_percent':100*float(gap),'within_one_percent':gap<=Q(1,100),'dual_support_constraints':len(weights),'dual_support_training_inputs':len(constants),'dual_constraint_kinds':{str(k):sum(x['kind']==k for x in cert['constraints']) for k in range(3)},'verified_primal_constraints':int(counts.sum()),'minimum_normalized_margins':list(map(float,minima)),'conservative_primal_allowance':float(allowance),'minimum_certified_margin':float(minima.min()-allowance),'original_linf':float(abs(initial).max()),'lower_over_original_linf':float(lower)/float(abs(initial).max()),'upper_over_original_linf':float(upper)/float(abs(initial).max()),'margin_tradeoff':tradeoff})
    assert len(records)==3
    return {'status':'passed','study_id':'H2x','decision':'all_three_optima_bounded_within_one_percent' if all(r['within_one_percent'] for r in records) else 'certified_intervals_with_reported_gaps','records':records,'total_verified_primal_constraints':sum(r['verified_primal_constraints'] for r in records),'total_exact_dual_support_rows':sum(r['dual_support_constraints'] for r in records),'seconds':time.perf_counter()-started,'verifier_sha256':V.sha(Path(__file__)),'primal_helper_sha256':V.sha(Path(V.__file__)),'protocol_sha256':V.sha(directory/'protocol.json'),'method':'Exactrational independentfullbinaryenergydifferences reconstruct eachdualsupportrow; nonnegativeweightedconstraints plus Hölder certify lowerbound withoutstationarity. Every feasibleprimal inequality is recomputed separately with conservativearithmeticallowance. Only publishedfiles and NumPy, no Torch/SciPy/featurebuilder/solver.','scope':'Minimum coordinateLinf norm in declared23parameterbasis and fixed unitmargin constraints on original1000trainingmazes/frozenreadouts. Coordinate/gauge/margin dependent; not an invariantstiffness, testaccuracy or globaloptimality claim.'}


if __name__=='__main__':
    root=Path(__file__).resolve().parents[3];p=argparse.ArgumentParser();p.add_argument('directory',nargs='?',type=Path,default=root/'results/research/work/training-min-norm');p.add_argument('training',nargs='?',type=Path,default=root/'results/research/work/contrastive-training');args=p.parse_args();print(json.dumps(verify(args.directory,args.training),indent=2))
