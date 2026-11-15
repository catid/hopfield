"""H2ad one bounded source-only LP/model with independent primal checks."""
import json
import signal
import time
import numpy as np
import study as S
import feasibility_features as F
import verify_training_feasibility as V
from mine_second_hard_negatives import PP,DEST,TRAIN,protocol


def check(theta,tokens,labels,attention,muoffs,aoffs,indices,boards,hamming):
    theta=theta.astype(V.LD);minimum=np.full(4,np.inf,V.LD);counts=np.zeros(4,np.int64)
    for lo in range(0,1000,64):
        t=tokens[lo:lo+64];y=labels[lo:lo+64];a=attention[lo:lo+64].astype(V.LD);free=t==1;canonical=V.energy(t,y,a,theta,muoffs,aoffs);on=(free&y).sum((1,2));off=(free&~y).sum((1,2));values=[V.flip(t,y,a,theta,muoffs,aoffs)[free],(V.energy(t,np.zeros_like(y),a,theta,muoffs,aoffs)-canonical)/on,(V.energy(t,np.ones_like(y),a,theta,muoffs,aoffs)-canonical)/off]
        for k,value in enumerate(values):minimum[k]=min(minimum[k],value.min());counts[k]+=value.size
    for lo in range(0,len(indices),64):
        ix=indices[lo:lo+64];t=tokens[ix];y=labels[ix];a=attention[ix].astype(V.LD);z=boards[lo:lo+64];value=(V.energy(t,z,a,theta,muoffs,aoffs)-V.energy(t,y,a,theta,muoffs,aoffs))/hamming[lo:lo+64];minimum[3]=min(minimum[3],value.min());counts[3]+=len(ix)
    norm=abs(theta).max();m=V.LD(256*900+1000);u=V.LD(2)**-53;gamma=m*u/(1-m*u);allowance=4*gamma*(10000*max(V.LD(1),norm)+1)
    return minimum,allowance,counts


def run():
    started=time.perf_counter();records=[];current=None;status='running'
    def alarm(signum,frame):raise TimeoutError('Frozen aggregate120sCPUwall envelope')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(120)
    try:
        import scipy
        from scipy.optimize import linprog
        from scipy.sparse import csr_matrix,hstack,vstack,eye
        p=protocol();mining=json.loads((DEST/'novelty-audit.json').read_text());assert mining['status']=='passed'
        with np.load(TRAIN/'training-arrays.npz') as z:tokens=z['tokens'];labels=z['labels']
        for seed in range(3):
            current=seed;folder=DEST/f'seed{seed}';folder.mkdir(exist_ok=True)
            if (folder/'result.json').exists():raise FileExistsError(folder)
            negative_path=DEST/f'all-negatives-seed{seed}.npz';assert S.sha(negative_path)==mining['records'][seed]['all_negative_archive_sha256']
            with np.load(negative_path) as z:indices=z['indices'];boards=z['boards'];hamming=z['hamming']
            record={'seed':seed,'mined_negatives':len(indices),'genuinely_new_boards':mining['records'][seed]['genuinely_new_boards'],'new_exact_constraint_rows':mining['records'][seed]['genuinely_new_exact_constraint_rows'],'previous_unit_margin_violations':mining['records'][seed]['previous_unit_margin_violations'],'protocol_sha256':S.sha(PP),'scipy':scipy.__version__,'numpy':np.__version__}
            if not mining['records'][seed]['genuinely_new_boards']:
                record.update(status='uninformative_no_new_board',aggregate_wall_seconds=time.perf_counter()-started);S.save(folder/'result.json',record);records.append(record);continue
            cp=S.OUT/f'training_feasibility/initial-seed{seed}.npz';assert S.sha(cp)==p['initial_models'][seed]['sha256']
            with np.load(cp) as z:muoffs=z['mu_offsets'];aoffs=z['aoffs']
            with np.load(TRAIN/f'train-readout-seed{seed}.npz') as z:attention=z['couplings']
            begin=time.perf_counter();matrices=[F.constraints(tokens[i],labels[i],attention[i],muoffs,aoffs)[0] for i in range(1000)];old=np.concatenate(matrices);del matrices
            new=np.stack([(F.energy_features(tokens[i],b,attention[i],muoffs,aoffs)-F.energy_features(tokens[i],labels[i],attention[i],muoffs,aoffs))/h for i,b,h in zip(indices,boards,hamming)])
            a=np.concatenate([old,new]);n=len(a);record.update(old_constraints=len(old),new_constraints=len(new),coefficient_seconds=time.perf_counter()-begin);S.emit('hard_negative_coefficients',**record)
            matrix=vstack([hstack([-csr_matrix(a),csr_matrix((n,1))]),hstack([eye(23),-np.ones((23,1))]),hstack([-eye(23),-np.ones((23,1))])],format='csr');rhs=np.concatenate([-np.ones(n),np.zeros(46)]);objective=np.zeros(24);objective[-1]=1.;remaining=120-(time.perf_counter()-started)
            if remaining<=0:raise TimeoutError('BudgetbeforeLP')
            begin=time.perf_counter();res=linprog(objective,A_ub=matrix,b_ub=rhs,bounds=[(None,None)]*23+[(0,None)],method='highs-ds',options=dict(p['solver']['options'],time_limit=remaining));record.update(solver_seconds=time.perf_counter()-begin,solver_status=int(res.status),solver_message=res.message,numerical_linf_optimum=float(res.x[-1]) if res.x is not None else None)
            if res.x is not None:np.savez_compressed(folder/'numerical-solution.npz',x=res.x,marginals=res.ineqlin.marginals if res.ineqlin.marginals is not None else np.array([]))
            if res.success:
                theta=np.asarray(2*res.x[:23],np.float64);np.save(folder/'candidate-theta.npy',theta);minimum,allowance,counts=check(theta,tokens,labels,attention,muoffs,aoffs,indices,boards,hamming);verified=bool(np.isfinite(minimum).all() and minimum.min()-allowance>1)
                record.update(status='verified_feasible_primal' if verified else 'unverified_primal',minimum_normalized_margins=list(map(float,minimum)),constraints_by_kind=counts.tolist(),conservative_absolute_allowance=float(allowance),certified_margin_lower=float(minimum.min()-allowance),candidate_linf=float(abs(theta).max()),candidate_sha256=S.sha(folder/'candidate-theta.npy'),minimum_float_feature_margin=float((a@theta).min()))
            else:record.update(status='numerical_status_only',scope='No independentlyverifiedinfeasibilitycertificate; numericalstatus cannot falsifyrealfeasibility.')
            record['aggregate_wall_seconds']=time.perf_counter()-started;S.save(folder/'result.json',record);records.append(record);S.emit('hard_negative_LP_model',**record)
        status='complete'
    except TimeoutError as e:status='budget_exhausted';S.emit('hard_negative_budget',current_seed=current,message=str(e))
    finally:
        signal.alarm(0);S.save(DEST/'solver-summary.json',{'status':status,'utc':S.utc(),'records':records,'current_seed':current,'aggregate_wall_seconds':time.perf_counter()-started,'protocol_sha256':S.sha(PP),'source_sha256':S.sha(__file__),'new_independent_test_inputs':0,'new_gradient_updates':0,'scope':'Source-onlyminednegativefeasibility; numericalLinfobjective withoutanoptimalitycertificate. Emptyminingseparate. No fresh test.'})


if __name__=='__main__':run()
