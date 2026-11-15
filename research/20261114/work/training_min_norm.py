"""Minimum coordinateLinf answer norm with certified primal/Hölder bounds."""
import argparse
from fractions import Fraction as Q
import json
import signal
import time
import numpy as np
import study as S
import feasibility_features as F
import verify_training_feasibility as V

DEST=S.OUT/'training_min_norm'
TRAIN=S.OUT/'contrastive_training'
PARENT=S.OUT/'training_feasibility'
PP=S.HERE/'training-min-norm-protocol.json'


def enc(q):return {'n':str(q.numerator),'d':str(q.denominator)}


def freeze():
    if PP.exists():raise FileExistsError(PP)
    parent=json.loads((S.HERE/'training-feasibility-protocol.json').read_text());gate=json.loads((PARENT/'independent-verification.json').read_text());assert gate['status']=='passed'
    p={'id':'source-only-minimum-coordinate-linf-answer-norm','study_id':'H2x','frozen_utc':S.utc(),'status':'frozen_before_min_norm_solver','question':'What minimum coordinateLinf norm is required by exactly the H2w unit-margin originaltrainingconstraints, and can independentupper/lower certificates bound it within1%relativewidth for everyfixedreadout?','constraints':parent['constraints'],'parameter_domain':parent['parameter_domain'],'objective':'minimize t with every original normalizedrow A_i theta>=1, -t<=theta_j<=t for all23coordinates and t>=0. t is optimized, not an artificialcoefficient bound.','interpretation':'Norm is coordinate/gauge/margin dependent. It is not an invariant dynamicalstiffness measure. No necessityclaim from a feasibleprimal alone; compare a certifiedlower bound with originalcoordinate norm only in this declared representation. No targetinputs or gradientfits.',
       'solver':{'method':'SciPyHiGHS dualsimplex, oneCPUthread','options':{'presolve':True,'threads':1,'small_matrix_value':1e-12,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9},'models':[0,1,2],'no_retuning':'One optimization call permodel, fixed constraints/objective/options. No parameterbound or alternative solver after outcomes.'},
       'primal_certificate':{'method':'Independent directNumPy extendedprecision pair/link energy and flip evaluation on every trainingconstraint. Same conservativefloat64allowance as H2w.','fixed_scaling':'Evaluate unscaled normalized minimum m and allowance e. If m-e>0, once set scale=nextafter(float64((1+1e-8)/(m-e)),+inf), candidate=float64(scale*theta). Recompute all inequalities; retain certificate only if newminimum-newallowance>1. No adaptive retries.','upper':'Maximumabsolute coordinate of the actual savedfloat64feasiblecandidate.'},
       'dual_certificate':{'weights':'Positive solverrow multipliers for original Arows only, each float64 interpreted as exactdyadicrational. Ignore boxrow multipliers.','coefficient_reconstruction':'Exactrational features from retainedfloat32readouts on every supportrow.','bound':'For any lambda>=0 with positive sum and nonzero g=A^Tlambda, every feasible theta has ||theta||inf>=sum(lambda)/||g||1 by weightedconstraints and Hölder. No exact dualstationarity assumption is needed.','audit':'Independent public-only rational reconstruction using directbinaryenergychanges plus exactweightedgradient, nonnegativity, numerator/denominator and ratio checks.'},
       'interval_rule':'Report certified lower L and feasible upper U. Narrow bound if (U-L)/U<=0.01; otherwise report gap without claiming approximateoptimality. A numericalsolver optimum is not itself thecertificate.',
       'budget':{'aggregate_cpu_wall_seconds':120,'threads':1,'includes':'Scientificrunentry imports/sourcechecks/data loading, all3coefficient extractions/solves/candidatechecks/certificates and receipts.','watchdog':'SIGALRM120s plus remaining HiGHStime_limit; preserve partialrecords, no extension. Independentread-onlycertificateaudit may follow, with no new optimization.'},
       'parent_protocol_sha256':S.sha(S.HERE/'training-feasibility-protocol.json'),'parent_verification_sha256':S.sha(PARENT/'independent-verification.json'),'training_manifest':parent['training_manifest'],'readouts':parent['readouts'],'initial_models':parent['initial_models'],'source_sha256':{str(path.relative_to(S.ROOT)):S.sha(path) for path in [S.HERE/'training_min_norm.py',S.HERE/'feasibility_features.py',S.HERE/'verify_training_feasibility.py',S.HERE/'study.py']}}
    DEST.mkdir(parents=True,exist_ok=True);S.save(PP,p);S.emit('training_min_norm_frozen',sha256=S.sha(PP))


def checked_margins(theta,tokens,labels,attention,muoffs,aoffs):
    theta=theta.astype(V.LD);minima=np.full(3,np.inf,V.LD);counts=np.zeros(3,np.int64)
    for lo in range(0,1000,64):
        t=tokens[lo:lo+64];y=labels[lo:lo+64];a=attention[lo:lo+64].astype(V.LD);free=t==1;on=(free&y).sum((1,2));off=(free&~y).sum((1,2));canonical=V.energy(t,y,a,theta,muoffs,aoffs);values=[V.flip(t,y,a,theta,muoffs,aoffs)[free],(V.energy(t,np.zeros_like(y),a,theta,muoffs,aoffs)-canonical)/on,(V.energy(t,np.ones_like(y),a,theta,muoffs,aoffs)-canonical)/off]
        for k,value in enumerate(values):assert np.isfinite(value).all();minima[k]=min(minima[k],value.min());counts[k]+=value.size
    norm=np.abs(theta).max();m=V.LD(256*900+1000);u=V.LD(2)**-53;gamma=m*u/(1-m*u);allowance=4*gamma*(10000*max(V.LD(1),norm)+1);return minima,allowance,counts


def run():
    started=time.perf_counter();records=[];status='running';current_seed=None
    def alarm(signum,frame):raise TimeoutError('Frozen120saggregateCPUwall envelope')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(120)
    try:
        import scipy
        from scipy.optimize import linprog
        from scipy.sparse import csr_matrix,hstack,vstack,eye
        p=json.loads(PP.read_text())
        for path,h in p['source_sha256'].items():assert S.sha(S.ROOT/path)==h
        assert S.sha(TRAIN/'training-arrays.npz')==p['training_manifest']['arrays_sha256']
        with np.load(TRAIN/'training-arrays.npz') as z:tokens=z['tokens'];labels=z['labels']
        for seed in range(3):
            current_seed=seed;folder=DEST/f'seed{seed}';folder.mkdir(exist_ok=True)
            if (folder/'result.json').exists():raise FileExistsError(folder)
            cp=PARENT/f'initial-seed{seed}.npz';assert S.sha(cp)==p['initial_models'][seed]['sha256']
            with np.load(cp) as z:muoffs=z['mu_offsets'];aoffs=z['aoffs'];initial=np.concatenate([z['T'].ravel(),z['b'],z['kappa']]).astype(np.float64)
            ap=TRAIN/f'train-readout-seed{seed}.npz';assert S.sha(ap)==p['readouts'][str(seed)]['readout_sha256']
            with np.load(ap) as z:attention=z['couplings']
            matrices=[];kinds=[];flats=[];indices=[];begin=time.perf_counter()
            for i in range(1000):
                a,k,f=F.constraints(tokens[i],labels[i],attention[i],muoffs,aoffs);matrices.append(a);kinds.append(k);flats.append(f);indices.append(np.full(len(a),i,np.int32))
            a=np.concatenate(matrices);kind=np.concatenate(kinds);flat=np.concatenate(flats);index=np.concatenate(indices);del matrices,kinds,flats,indices;n=len(a);record={'seed':seed,'status':'unverified','rows':n,'coefficient_seconds':time.perf_counter()-begin,'initial_parameter_linf':float(abs(initial).max()),'protocol_sha256':S.sha(PP),'scipy':scipy.__version__,'numpy':np.__version__};S.emit('min_norm_coefficients',**record)
            matrix=vstack([hstack([-csr_matrix(a),csr_matrix((n,1))]),hstack([eye(23),-np.ones((23,1))]),hstack([-eye(23),-np.ones((23,1))])],format='csr');b=np.concatenate([-np.ones(n),np.zeros(46)]);objective=np.zeros(24);objective[-1]=1.;remaining=120-(time.perf_counter()-started)
            if remaining<=0:raise TimeoutError('BudgetbeforeminnormLP')
            begin=time.perf_counter();res=linprog(objective,A_ub=matrix,b_ub=b,bounds=[(None,None)]*23+[(0,None)],method='highs-ds',options=dict(p['solver']['options'],time_limit=remaining));record.update(solver_seconds=time.perf_counter()-begin,solver_status=int(res.status),solver_message=res.message,numerical_optimum=float(res.x[-1]) if res.x is not None else None)
            if res.x is not None:np.savez_compressed(folder/'numerical-solution.npz',x=res.x,marginals=res.ineqlin.marginals if res.ineqlin.marginals is not None else np.array([]))
            if res.success:
                raw=res.x[:23];rawmin,rawe,counts=checked_margins(raw,tokens,labels,attention,muoffs,aoffs);minimum=rawmin.min();record.update(raw_minimum_margins=list(map(float,rawmin)),raw_allowance=float(rawe));upper=None
                if minimum-rawe>0:
                    scale=float(np.nextafter(np.float64((1+V.LD('1e-8'))/(minimum-rawe)),np.inf));theta=np.asarray(scale*raw,np.float64);np.save(folder/'candidate-theta.npy',theta);margins,allowance,counts=checked_margins(theta,tokens,labels,attention,muoffs,aoffs);certified=margins.min()-allowance>1
                    record.update(scale=scale,minimum_source_margin_by_kind=list(map(float,margins)),conservative_absolute_allowance=float(allowance),certified_margin_lower=float(margins.min()-allowance),primal_verified=bool(certified),candidate_sha256=S.sha(folder/'candidate-theta.npy'),parameter_l1=float(abs(theta).sum()),parameter_l2=float(np.linalg.norm(theta)))
                    if certified:upper=Q.from_float(float(abs(theta).max()));record['upper_bound']=float(upper)
                weights=-res.ineqlin.marginals[:n];support=np.flatnonzero(weights>0);metadata=[{'constraint_row':int(j),'training_index':int(index[j]),'kind':int(kind[j]),'flat_index':int(flat[j])} for j in support];exact=[F.exact_row(tokens[r['training_index']],labels[r['training_index']],attention[r['training_index']],muoffs,aoffs,r['kind'],r['flat_index']) for r in metadata];lambdas=[Q.from_float(float(weights[j])) for j in support];gradient=[sum((weight*row[k] for weight,row in zip(lambdas,exact)),Q(0)) for k in range(23)];numerator=sum(lambdas,Q(0));denominator=sum(map(abs,gradient),Q(0));assert numerator>0 and denominator>0;lower=numerator/denominator;constants=[]
                for i in sorted({r['training_index'] for r in metadata}):
                    path=folder/f'train{i:04d}-constants.npz';np.savez_compressed(path,tokens=tokens[i],label=labels[i],attention=attention[i],mu_offsets=muoffs,aoffs=aoffs);constants.append({'training_index':i,'path':path.name,'sha256':S.sha(path)})
                certificate={'kind':'NonnegativeweightedconstraintHölderbound','constraints':metadata,'rows':[[enc(v) for v in row] for row in exact],'weights':[enc(v) for v in lambdas],'gradient':[enc(v) for v in gradient],'numerator':enc(numerator),'denominator':enc(denominator),'lower_bound':enc(lower),'upper_bound':enc(upper) if upper is not None else None,'constants':constants,'scope':'For anyfeasibletheta, sumlambda<=lambda^TA theta<=||theta||inf||A^Tlambda||1. Exact nonnegativeweights/coefficients suffice; no dualstationarityassumption.'};S.save(folder/'norm-certificate.json',certificate);record.update(lower_bound=float(lower),dual_support=len(support),certificate_sha256=S.sha(folder/'norm-certificate.json'))
                if upper is not None:
                    assert lower<=upper;gap=(upper-lower)/upper;record.update(relative_certified_gap=float(gap),status='narrow_certified_interval_candidate' if gap<=Q(1,100) else 'certified_interval_candidate',lower_over_original_linf=float(lower)/record['initial_parameter_linf'],upper_over_original_linf=float(upper)/record['initial_parameter_linf'])
                else:record.update(status='lower_bound_only_candidate')
            else:record.update(status='unresolved_solver')
            record['aggregate_wall_seconds']=time.perf_counter()-started;S.save(folder/'result.json',record);records.append(record);S.emit('training_min_norm_model',**record)
        status='complete'
    except TimeoutError as e:status='budget_exhausted';S.emit('training_min_norm_budget',current_seed=current_seed,message=str(e))
    finally:
        signal.alarm(0);S.save(DEST/'solver-summary.json',{'status':status,'utc':S.utc(),'records':records,'current_seed':current_seed,'aggregate_wall_seconds':time.perf_counter()-started,'protocol_sha256':S.sha(PP),'script_sha256':S.sha(__file__),'new_training_or_inference':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','run']);args=parser.parse_args();globals()[args.command]()
