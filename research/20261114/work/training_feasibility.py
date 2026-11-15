"""Source-only23parameter linear feasibility; all3models share120CPUwallseconds."""
import argparse
from fractions import Fraction as Q
import json
import math
import os
import signal
import sys
import time
import numpy as np
import study as S
import feasibility_features as F

DEST=S.OUT/'training_feasibility'
TRAIN=S.OUT/'contrastive_training'
PP=S.HERE/'training-feasibility-protocol.json'


def encode(q):return {'n':str(q.numerator),'d':str(q.denominator)}


def freeze():
    if PP.exists():raise FileExistsError(PP)
    gate=json.loads((DEST/'feature-gate.json').read_text());assert gate['status']=='passed' and gate['feature_sha256']==S.sha(F.__file__)
    p={'id':'source-only-answer-energy-linear-feasibility','hypothesis':'H2w: for each of the3fixedoriginal trainingreadouts, some real23parameter answerenergy satisfies all unitnormalizedranking margins and every unitlocalflip margin on the1000originaltrainingmazes.','frozen_utc':S.utc(),'status':'frozen_before_any_LP','parameter_domain':{'order':'T[4,2,2] rowmajor, b[2], kappa[5]','dimension':23,'domain':'All real23vectors; no coefficientbound, signrestriction, gaugefix or proximitypenalty. Exactly the trainedarchitecture answerparameters.','frozen':'Original mu,beta and exact retainedfloat32couplings, promoted to float64.'},'constraints':[{'kind':0,'meaning':'Every freecell i of each canonicaltraininglabel y','signed_inequality':'E(y with i flipped)-E(y)>=1','normalization':1},{'kind':1,'meaning':'Sparse negative: endpoints only, clamped walls/endpoints','signed_inequality':'E(endpoints)-E(y)>=number of selectedfree labelcells','normalization':'Divide by selectedfree labelcount, assertedpositive.'},{'kind':2,'meaning':'Dense negative: all open cells, clamped walls/endpoints','signed_inequality':'E(allopen)-E(y)>=number of unselectedfree cells','normalization':'Divide by unselectedfree count, assertedpositive.'}], 'decision':'Support only with independently verified feasibleprimal for all3models. Falsify if an exact independently verified nonnegativeFarkas certificate proves at leastone modelinfeasible. Numericalinfeasibility, numericalresidual, timeout or uncertifiedhugeweights alone are inconclusive. No claim beyond original1000trainingarrays/fixedreadouts; no targetdata, newfit or inference.',
       'solver':{'method':'SciPyHiGHS dualsimplex, oneCPUthread; elasticfeasibility minimize s>=0 subject A*theta+s>=1. theta unbounded.','fixed_options':{'presolve':True,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'small_matrix_value':1e-12,'threads':1},'interpretation':'LPfloatcoefficients may discard tinyentries. Only independently verifiedprimal or exactFarkas proof determines mathematicalfeasibility.','feasible_candidate':'If elastic s<=1e-7, scale the returnedtheta by exactly2 before independent sourceverification, exploiting homogeneousconstraints; no search for other scale or normbound.','dual_candidate':'Use positive dualsupport >1e-10, reconstruct supportrowcoefficients as exactrationals from storedfloat32readouts, solve exact stationarity plus sumweights1 by rationalRREF. Retain failure as uncertified, not infeasible.'},
       'budget':{'aggregate_cpu_wall_seconds':120,'models':[0,1,2],'threads':1,'includes':'After scientificrunentry: imports, sourcehashes, data loading, coefficientextraction, all3LPs, candidatechecks/certificates and receipts. Dependencies/numerical no-LPgate prepared beforehand.','watchdog':'SIGALRM120s; HiGHStime_limit receives remainingtime. Preserve completedrecords/timeout state; no extension or secondsolver after outcomes. A separate read-only independent audit may verify retained certificates; no new numericaloptimization.'},
       'primal_verification':{'method':'Independent upstreamfloat64energy and flip_delta on every originaltrainingmaze/freecell, compared to separate directlinear features.','allowance':'m=256*900+1000, u=2^-53, gamma=m*u/(1-m*u); commonabsoluteallowance=4*gamma*(10000*max(1,||theta||inf)+1). This upperbounds bothbinaryenergycomparisons and individualfliparithmetic; normalizedgaps divide by positiveintegercounts>=1. Require every source normalizedmargin>=1+allowance and source/feature residual<=allowance.','norms':'Record maximumabsolute, L1 and L2parameter norms, difference from initialtheta, minimum source margins and coefficientmagnitude ranges.'},
       'training_manifest':json.loads((TRAIN/'training-manifest.json').read_text()),'training_manifest_sha256':S.sha(TRAIN/'training-manifest.json'),'readouts':{str(seed):json.loads((TRAIN/f'train-readout-seed{seed}.json').read_text()) for seed in range(3)},'initial_models':json.loads((DEST/'initial-manifest.json').read_text()),'gate':{'path':str((DEST/'feature-gate.json').relative_to(S.ROOT)),'sha256':S.sha(DEST/'feature-gate.json')},'source_sha256':{str(path.relative_to(S.ROOT)):S.sha(path) for path in [S.HERE/'training_feasibility.py',S.HERE/'feasibility_features.py',S.HERE/'gate_feasibility.py',S.HERE/'study.py',S.UP/'cram/cram_maze_hard_graph.py',S.UP/'cram/maze_h3.py',S.UP/'cram/maze_learned.py']}}
    S.save(PP,p);S.emit('training_feasibility_frozen',sha256=S.sha(PP))


def rational_solution(rows,raw):
    """Exact rational stationarity with approximatefreevalues only as parameters."""
    n=len(rows);matrix=[[rows[j][i] for j in range(n)]+[Q(0)] for i in range(23)]+[[Q(1)]*n+[Q(1)]];pivots=[];r=0
    for c in range(n):
        pivot=next((k for k in range(r,len(matrix)) if matrix[k][c]),None)
        if pivot is None:continue
        matrix[r],matrix[pivot]=matrix[pivot],matrix[r];v=matrix[r][c];matrix[r]=[x/v for x in matrix[r]]
        for k in range(len(matrix)):
            if k!=r and matrix[k][c]:
                factor=matrix[k][c];matrix[k]=[x-factor*y for x,y in zip(matrix[k],matrix[r])]
        pivots.append((r,c));r+=1
        if r==len(matrix):break
    if any(not any(row[:-1]) and row[-1] for row in matrix):return None
    chosen={c for _,c in pivots};free=[c for c in range(n) if c not in chosen];x=[Q(0)]*n;normalizer=sum(raw)
    for c in free:x[c]=Q.from_float(float(raw[c]/normalizer))
    for r,c in pivots:x[c]=matrix[r][-1]-sum((matrix[r][j]*x[j] for j in free),Q(0))
    if not all(v>=0 for v in x) or sum(x,Q(0))!=1:return None
    assert all(sum((weight*row[k] for weight,row in zip(x,rows)),Q(0))==0 for k in range(23));return x


def run():
    started=time.perf_counter();records=[];DEST.mkdir(parents=True,exist_ok=True)
    def alarm(signum,frame):raise TimeoutError('Frozen aggregate120sCPUwall envelope')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(120);status='running';current_seed=None
    try:
        sys.path.append(str(S.ROOT/'artifacts/feasibility-venv/lib/python3.12/site-packages'))
        import scipy
        from scipy.optimize import linprog
        from scipy.sparse import csr_matrix,hstack
        torch,_,_=S.imports();p=json.loads(PP.read_text())
        for path,h in p['source_sha256'].items():assert S.sha(S.ROOT/path)==h
        assert S.sha(TRAIN/'training-arrays.npz')==p['training_manifest']['arrays_sha256']
        with np.load(TRAIN/'training-arrays.npz') as z:tokens=z['tokens'];labels=z['labels']
        for seed in range(3):
            current_seed=seed;folder=DEST/f'seed{seed}';folder.mkdir(exist_ok=True)
            if (folder/'result.json').exists():raise FileExistsError(folder)
            remaining=120-(time.perf_counter()-started)
            if remaining<=0:raise TimeoutError('Frozenbudgetbeforemodel')
            cp=DEST/f'initial-seed{seed}.npz';assert S.sha(cp)==p['initial_models'][seed]['sha256']
            with np.load(cp) as z:muoffs=z['mu_offsets'];aoffs=z['aoffs'];initial=np.concatenate([z['T'].ravel(),z['b'],z['kappa']]).astype(np.float64)
            ap=TRAIN/f'train-readout-seed{seed}.npz';assert S.sha(ap)==p['readouts'][str(seed)]['readout_sha256']
            with np.load(ap) as z:attention=z['couplings']
            matrices=[];kinds=[];flats=[];inputs=[];extract=time.perf_counter()
            for i in range(1000):
                a,k,f=F.constraints(tokens[i],labels[i],attention[i],muoffs,aoffs);matrices.append(a);kinds.append(k);flats.append(f);inputs.append(np.full(len(a),i,np.int32))
            matrix=np.concatenate(matrices);kind=np.concatenate(kinds);flat=np.concatenate(flats);indices=np.concatenate(inputs);del matrices,kinds,flats,inputs
            nonzero=np.abs(matrix[matrix!=0]);record={'seed':seed,'status':'unverified','rows':len(matrix),'localflip_constraints':int((kind==0).sum()),'sparse_constraints':1000,'dense_constraints':1000,'feature_seconds':time.perf_counter()-extract,'coefficient_minimum_nonzero_abs':float(nonzero.min()),'coefficient_maximum_abs':float(nonzero.max()),'coefficient_frobenius_norm':float(np.linalg.norm(matrix)),'initial_parameter_linf':float(abs(initial).max()),'protocol_sha256':S.sha(PP),'scipy':scipy.__version__,'numpy':np.__version__};S.emit('feasibility_coefficients',**record)
            remaining=120-(time.perf_counter()-started)
            if remaining<=0:raise TimeoutError('FrozenbudgetbeforeLP')
            lp=hstack([-csr_matrix(matrix),-np.ones((len(matrix),1))],format='csr');objective=np.zeros(24);objective[-1]=1.;lp_start=time.perf_counter();res=linprog(objective,A_ub=lp,b_ub=-np.ones(len(matrix)),bounds=[(None,None)]*23+[(0,None)],method='highs-ds',options=dict(p['solver']['fixed_options'],time_limit=remaining));record.update(solver_seconds=time.perf_counter()-lp_start,solver_status=int(res.status),solver_message=res.message,elastic_slack=float(res.x[-1]) if res.x is not None else None)
            if res.x is not None:np.savez_compressed(folder/'numerical-solution.npz',x=res.x,marginals=res.ineqlin.marginals if res.ineqlin.marginals is not None else np.array([]))
            if res.success and res.x[-1]<=1e-7:
                theta=2*res.x[:23];np.save(folder/'candidate-theta.npy',theta);model=S.model_for(seed,30,'original','cpu',torch.float64)
                with torch.no_grad():model.T.copy_(torch.tensor(theta[:16].reshape(4,2,2)));model.b.copy_(torch.tensor(theta[16:18]));model.kappa.copy_(torch.tensor(theta[18:]))
                values=[]
                with torch.no_grad():
                    for i in range(1000):
                        t=torch.tensor(tokens[i].astype(np.int64))[None];y=torch.tensor(labels[i],dtype=torch.float64)[None];a=torch.tensor(attention[i],dtype=torch.float64)[None];free=t==1;on=int((free&y.bool()).sum());off=int((free&~y.bool()).sum());energy=model.energy(t,y,a);flip=model.flip_delta(t,y,a)[free].numpy();ep=float((model.energy(t,torch.zeros_like(y),a)-energy)/on);dense=float((model.energy(t,torch.ones_like(y),a)-energy)/off);values.append(np.concatenate([flip,[ep,dense]]))
                values=np.concatenate(values);predicted=matrix@theta;norm=float(abs(theta).max());m=256*900+1000;u=2.**-53;gamma=m*u/(1-m*u);allowance=4*gamma*(10000*max(1.,norm)+1);residual=float(np.max(abs(values-predicted)));minimum=float(values.min());verified=minimum>=1+allowance and residual<=allowance
                record.update(status='verified_feasible_primal' if verified else 'unverified_primal',parameter_linf=norm,parameter_l1=float(abs(theta).sum()),parameter_l2=float(np.linalg.norm(theta)),change_from_initial_linf=float(abs(theta-initial).max()),minimum_source_normalized_margin=minimum,minimum_source_margin_by_kind={str(k):float(values[kind==k].min()) for k in range(3)},maximum_source_feature_residual=residual,conservative_absolute_allowance=allowance,candidate_sha256=S.sha(folder/'candidate-theta.npy'))
            elif res.success and res.ineqlin.marginals is not None:
                dual=-res.ineqlin.marginals;support=np.flatnonzero(dual>1e-10);metadata=[{'constraint_row':int(j),'training_index':int(indices[j]),'kind':int(kind[j]),'flat_index':int(flat[j])} for j in support];exact=[F.exact_row(tokens[r['training_index']],labels[r['training_index']],attention[r['training_index']],muoffs,aoffs,r['kind'],r['flat_index']) for r in metadata];weights=rational_solution(exact,dual[support]);record.update(dual_support=len(support),maximum_float_stationarity_residual=float(np.max(abs(matrix[support].T@dual[support]))))
                if weights is not None:
                    constants=[]
                    for i in sorted({r['training_index'] for r in metadata}):
                        path=folder/f'train{i:04d}-constants.npz';np.savez_compressed(path,tokens=tokens[i],label=labels[i],attention=attention[i],mu_offsets=muoffs,aoffs=aoffs);constants.append({'training_index':i,'path':path.name,'sha256':S.sha(path)})
                    certificate={'kind':'ExactFarkas','constraints':metadata,'rows':[[encode(v) for v in row] for row in exact],'weights':[encode(v) for v in weights],'rhs_each':1,'exact_weight_sum':encode(sum(weights,Q(0))),'exact_stationarity':[encode(sum((weight*row[k] for weight,row in zip(weights,exact)),Q(0))) for k in range(23)],'constants':constants,'scope':'Unrestrictedreal23parameter architecture. Nonnegativeweights, sum1 and exactweightedrowzero contradict everyrow dot theta>=1.'};S.save(folder/'farkas-certificate.json',certificate);record.update(status='exact_farkas_candidate',certificate_sha256=S.sha(folder/'farkas-certificate.json'))
                else:record.update(status='numerically_infeasible_only',reason='NoexactnonnegativeFarkas certificate recovered; tinycoefficients or illconditioning maymatter.')
            else:record.update(status='unresolved_solver',reason='Numericalsolverfailure is not mathematicalinfeasibility.')
            record['aggregate_wall_seconds']=time.perf_counter()-started;S.save(folder/'result.json',record);records.append(record);S.emit('training_feasibility_model',**record)
        status='complete'
    except TimeoutError as e:
        status='budget_exhausted';S.emit('training_feasibility_budget',current_seed=current_seed,message=str(e))
    finally:
        signal.alarm(0);S.save(DEST/'solver-summary.json',{'status':status,'utc':S.utc(),'records':records,'current_seed':current_seed,'aggregate_wall_seconds':time.perf_counter()-started,'protocol_sha256':S.sha(PP),'script_sha256':S.sha(__file__),'new_training_or_inference':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','run']);args=parser.parse_args();globals()[args.command]()
