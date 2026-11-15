"""H2ag bounded primal LP with one conditional checkable Farkas attempt."""
from fractions import Fraction as Q
import json
import signal
import time
import numpy as np
import study as S
import feasibility_features as F
import verify_training_feasibility as V
from verify_second_novelty import feature
from solve_hard_negatives import check
from leaf_block_feasibility import PP,DEST,SOURCE,TRAIN,PRIOR,protocol


def rational_solution(rows):
    k=len(rows);matrix=[[rows[j][i] for j in range(k)]+[Q(0)] for i in range(23)]+[[Q(1)]*k+[Q(1)]];pivot=[];r=0
    for c in range(k):
        found=next((j for j in range(r,len(matrix)) if matrix[j][c]),None)
        if found is None:continue
        matrix[r],matrix[found]=matrix[found],matrix[r];v=matrix[r][c];matrix[r]=[x/v for x in matrix[r]]
        for j in range(len(matrix)):
            if j!=r and matrix[j][c]:v=matrix[j][c];matrix[j]=[x-v*y for x,y in zip(matrix[j],matrix[r])]
        pivot.append((r,c));r+=1
        if r==len(matrix):break
    if any(not any(row[:-1]) and row[-1] for row in matrix):return None
    weights=[Q(0)]*k
    for r,c in pivot:weights[c]=matrix[r][-1]
    if not all(w>=0 for w in weights) or sum(weights)!=1 or any(sum(w*row[i] for w,row in zip(weights,rows)) for i in range(23)):return None
    return weights


def run():
    started=time.perf_counter();records=[];current=None;status='running'
    def alarm(signum,frame):raise TimeoutError('Frozen aggregate120CPU-second envelope')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(120)
    try:
        import scipy
        from scipy.optimize import linprog
        from scipy.sparse import csr_matrix,hstack,vstack,eye
        p=protocol();selection=json.loads((DEST/'move-selection.json').read_text());assert S.sha(DEST/'move-selection.json')==p['selection_sha256']
        with np.load(TRAIN/'training-arrays.npz') as z:tokens=z['tokens'];labels=z['labels']
        for seed in range(3):
            current=seed;folder=DEST/f'seed{seed}';folder.mkdir(exist_ok=True)
            if (folder/'result.json').exists():raise FileExistsError(folder)
            selected=selection['records'][seed];record={'seed':seed,'protocol_sha256':S.sha(PP),'eligible_leaves':selected['eligible_leaves'],'wrong_boards':selected['wrong_boards'],'scipy':scipy.__version__,'numpy':np.__version__}
            if not selected['informative']:
                record.update(status='uninformative_empty_or_no_leaf',aggregate_wall_seconds=time.perf_counter()-started);S.save(folder/'result.json',record);records.append(record);continue
            with np.load(PRIOR/f'all-negatives-seed{seed}.npz') as z:indices=z['indices'];boards=z['boards'];hamming=z['hamming']
            with np.load(SOURCE/f'negatives-seed{seed}.npz') as z:current_indices=z['indices'];current_boards=z['boards']
            with np.load(S.ROOT/p['initial_models'][seed]['path']) as z:mu=z['mu_offsets'];ao=z['aoffs']
            with np.load(TRAIN/f'train-readout-seed{seed}.npz') as z:attention=z['couplings']
            matrices=[F.constraints(tokens[i],labels[i],attention[i],mu,ao)[0] for i in range(1000)];ends=np.cumsum([len(x) for x in matrices]);base=np.concatenate(matrices);del matrices
            rank=np.stack([(F.energy_features(tokens[i],b,attention[i],mu,ao)-F.energy_features(tokens[i],labels[i],attention[i],mu,ao))/h for i,b,h in zip(indices,boards,hamming)])
            leaves=[]
            for item in selected['mapping']:
                i=item['training_index'];b=current_boards[item['negative_row']];rows,flat=F.flip_features(tokens[i],b,attention[i],mu,ao);j=int(np.flatnonzero(flat==item['flat'])[0]);leaves.append(-rows[j])
            blocks=[]
            for item in selected['blocks']:
                i=item['training_index'];b=current_boards[item['negative_row']];z=b.copy()
                for rr,cc in item['cells']:z[rr,cc]=False
                blocks.append((F.energy_features(tokens[i],b,attention[i],mu,ao)-F.energy_features(tokens[i],z,attention[i],mu,ao))/item['size'])
            a=np.concatenate([base,rank,np.stack(leaves),np.stack(blocks)]);n=len(a);record.update(old_base_rows=len(base),old_negative_rows=len(rank),new_leaf_rows=len(leaves),duplicate_component_rows=len(blocks))
            matrix=vstack([hstack([-csr_matrix(a),csr_matrix((n,1))]),hstack([eye(23),-np.ones((23,1))]),hstack([-eye(23),-np.ones((23,1))])],format='csr');rhs=np.concatenate([-np.ones(n),np.zeros(46)]);objective=np.zeros(24);objective[-1]=1.
            def remaining():
                value=120-(time.perf_counter()-started)
                if value<=0:raise TimeoutError('Budget exhausted')
                return value
            before=time.perf_counter();res=linprog(objective,A_ub=matrix,b_ub=rhs,bounds=[(None,None)]*23+[(0,None)],method='highs-ds',options=dict(p['solver']['options'],time_limit=remaining()));record.update(solver_seconds=time.perf_counter()-before,solver_status=int(res.status),solver_message=res.message)
            if res.x is not None:np.savez_compressed(folder/'numerical-solution.npz',x=res.x)
            if res.success:
                theta=np.asarray(2*res.x[:23],np.float64);np.save(folder/'candidate-theta.npy',theta);minimum,allowance,counts=check(theta,tokens,labels,attention,mu,ao,indices,boards,hamming);lm=np.inf
                for item in selected['mapping']:
                    i=item['training_index'];b=current_boards[item['negative_row']];delta=V.flip(tokens[i:i+1],b[None],attention[i:i+1].astype(V.LD),theta.astype(V.LD),mu,ao)[0];lm=min(lm,-delta.flat[item['flat']])
                bm=np.inf
                for item in selected['blocks']:
                    i=item['training_index'];b=current_boards[item['negative_row']];z=b.copy()
                    for rr,cc in item['cells']:z[rr,cc]=False
                    aa=attention[i:i+1].astype(V.LD);tt=tokens[i:i+1];th=theta.astype(V.LD);value=(V.energy(tt,b[None],aa,th,mu,ao)-V.energy(tt,z[None],aa,th,mu,ao))[0]/item['size'];bm=min(bm,value)
                minimum=np.concatenate([minimum,[lm,bm]]);counts=np.concatenate([counts,[len(leaves),len(blocks)]]);verified=bool(np.isfinite(minimum).all() and minimum.min()-allowance>1);record.update(status='verified_feasible_primal' if verified else 'unverified_primal',minimum_margins=list(map(float,minimum)),constraints_by_kind=counts.tolist(),conservative_absolute_allowance=float(allowance),certified_margin_lower=float(minimum.min()-allowance),candidate_linf=float(abs(theta).max()),candidate_sha256=S.sha(folder/'candidate-theta.npy'))
            elif res.status==2:
                before=time.perf_counter();dual=linprog(np.zeros(n),A_eq=vstack([csr_matrix(a.T),csr_matrix(np.ones((1,n)))],format='csr'),b_eq=np.concatenate([np.zeros(23),[1.]]),bounds=(0,None),method='highs-ds',options=dict(p['solver']['options'],time_limit=remaining()));record.update(dual_seconds=time.perf_counter()-before,dual_status=int(dual.status),status='numerical_infeasibility_only');exact_rows=[];descriptors=[]
                if dual.success:
                    support=np.flatnonzero(dual.x>0);np.savez_compressed(folder/'numerical-dual.npz',support=support,weights=dual.x[support])
                    for j in support:
                        if j<len(base):
                            i=int(np.searchsorted(ends,j,side='right'));start=0 if i==0 else int(ends[i-1]);off=int(j)-start;free=np.flatnonzero(tokens[i]==1);y=labels[i];z=y.copy();h=1
                            if off<len(free):f=int(free[off]);z.flat[f]=~z.flat[f];desc={'kind':'canonical_flip','input_index':i,'flat':f}
                            else:
                                kind=off-len(free);z=np.zeros_like(y) if kind==0 else np.ones_like(y);h=int(((tokens[i]==1)&(y if kind==0 else ~y)).sum());desc={'kind':'endpoint' if kind==0 else 'allopen','input_index':i,'hamming':h}
                            row=tuple((v-w)/h for v,w in zip(feature(tokens[i],z,attention[i],mu,ao),feature(tokens[i],y,attention[i],mu,ao)))
                        elif j<len(base)+len(rank):
                            k=int(j)-len(base);i=int(indices[k]);h=int(hamming[k]);row=tuple((v-w)/h for v,w in zip(feature(tokens[i],boards[k],attention[i],mu,ao),feature(tokens[i],labels[i],attention[i],mu,ao)));desc={'kind':'old_negative','negative_row':k,'input_index':i,'hamming':h}
                        elif j<len(base)+len(rank)+len(leaves):
                            k=int(j)-len(base)-len(rank);item=selected['mapping'][k];i=item['training_index'];z=current_boards[item['negative_row']].copy();z.flat[item['flat']]=False;row=tuple(v-w for v,w in zip(feature(tokens[i],current_boards[item['negative_row']],attention[i],mu,ao),feature(tokens[i],z,attention[i],mu,ao)));desc={'kind':'leaf_removal','leaf_row':k,**item}
                        else:
                            k=int(j)-len(base)-len(rank)-len(leaves);item=selected['blocks'][k];i=item['training_index'];b=current_boards[item['negative_row']];z=b.copy()
                            for rr,cc in item['cells']:z[rr,cc]=False
                            row=tuple((v-w)/item['size'] for v,w in zip(feature(tokens[i],b,attention[i],mu,ao),feature(tokens[i],z,attention[i],mu,ao)));desc={'kind':'component_removal',**item}
                        exact_rows.append(row);descriptors.append(desc)
                    weights=rational_solution(exact_rows)
                    if weights is not None:
                        certificate={'seed':seed,'protocol_sha256':S.sha(PP),'rows':[{'descriptor':desc,'coefficients':[[str(x.numerator),str(x.denominator)] for x in row],'weight':[str(w.numerator),str(w.denominator)]} for desc,row,w in zip(descriptors,exact_rows,weights)],'claim':'All source inequalities have form row dot theta>=1; nonnegative rational weights sum1 and weighted row sum is exactly0, hence0>=1 contradiction.'};S.save(folder/'farkas-certificate.json',certificate);record.update(status='exact_farkas_infeasible',farkas_sha256=S.sha(folder/'farkas-certificate.json'),support_rows=len(weights))
            else:record.update(status='numerical_status_only')
            record['aggregate_wall_seconds']=time.perf_counter()-started;S.save(folder/'result.json',record);records.append(record);S.emit('leaf_block_model',**record)
        status='complete'
    except TimeoutError as error:status='budget_exhausted';S.emit('leaf_block_budget',current_seed=current,message=str(error))
    finally:
        signal.alarm(0);S.save(DEST/'solver-summary.json',{'status':status,'utc':S.utc(),'records':records,'current_seed':current,'aggregate_wall_seconds':time.perf_counter()-started,'protocol_sha256':S.sha(PP),'source_sha256':S.sha(__file__),'new_independent_test_inputs':0,'scope':'Every prior source row and every selected leaf retained. Feasible primal or exact Farkas certificate independently auditable. Numerical statuses alone are inconclusive. No fresh target inference.'})


if __name__=='__main__':run()
