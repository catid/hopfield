"""Secondary LP allocation; convert primal/dual to independently checkable rationals."""
from fractions import Fraction as Q
import json
import time
import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
import study as S

DEST=S.OUT/'global_bound'


def decode(d):return Q(int(d['numerator']),int(d['denominator']))
def encode(q):return {'numerator':str(q.numerator),'denominator':str(q.denominator)}


def main():
    records=[];started=time.perf_counter()
    for seed in range(3):
        out=DEST/f'seed{seed}';half=json.loads((out/'half-result.json').read_text());c=json.loads((out/'coefficients.json').read_text());assert half['coefficients_sha256']==S.sha(out/'coefficients.json')
        if half['status']=='certified':records.append({'seed':seed,'status':'not_needed_half_certified'});continue
        denominator=int(c['common_denominator']);h=[Q(int(x),denominator) for x in c['unary_numerators']];negative=[(tuple(pair),Q(-int(x),denominator)) for pair,x in zip(c['edge_nodes'],c['edge_numerators']) if int(x)<0];n=len(h);m=len(negative);rows=[];cols=[];values=[];b=np.array([float(x) for x in h]);w=[]
        for e,((i,j),weight) in enumerate(negative):rows.extend([i,j]);cols.extend([e,e]);values.extend([1.,-1.]);b[j]-=float(weight);w.append(float(weight))
        rows.extend(range(n));cols.extend([m]*n);values.extend([1.]*n);matrix=coo_matrix((values,(rows,cols)),shape=(n,m+1)).tocsr();objective=np.zeros(m+1);objective[-1]=-1.;begin=time.perf_counter();result=linprog(objective,A_ub=matrix,b_ub=b,bounds=[(0.,x) for x in w]+[(None,None)],method='highs',options={'time_limit':60.,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9});seconds=time.perf_counter()-begin
        record={'seed':seed,'solver_status':int(result.status),'solver_message':result.message,'solver_seconds':seconds,'scipy':scipy.__version__,'coefficients_sha256':S.sha(out/'coefficients.json'),'script_sha256':S.sha(__file__),'secondary_predeclared':True}
        if not result.success:
            record.update(status='no_verified_certificate',scope='UnsuccessfulLP is not a globaloptimum conclusion.');S.save(out/'allocation-result.json',record);records.append(record);continue
        allocation=[min(weight,max(Q(0),Q.from_float(float(x)))) for x,(_,weight) in zip(result.x[:m],negative)];loads=[Q(0)]*n
        for ((i,j),weight),x in zip(negative,allocation):loads[i]+=x;loads[j]+=weight-x
        slack=[value-load for value,load in zip(h,loads)];lower=min(slack);raw_dual=[max(Q(0),Q.from_float(float(-x))) for x in result.ineqlin.marginals];normalizer=sum(raw_dual,Q(0));assert normalizer>0;dual=[x/normalizer for x in raw_dual]
        upper=sum((y*x for y,x in zip(dual,h)),Q(0))-sum((weight*min(dual[i],dual[j]) for (i,j),weight in negative),Q(0));assert sum(dual,Q(0))==1 and all(x>=0 for x in dual) and lower<=upper
        allowance=decode(c['roundoff_allowance']);certificate={'negative_edge_order':[list(pair) for pair,_ in negative],'edge_allocation_to_first_vertex':[encode(x) for x in allocation],'node_slacks':[encode(x) for x in slack],'normalized_nonnegative_dual':[encode(x) for x in dual],'exact_primal_lower':encode(lower),'exact_dual_upper':encode(upper),'exact_duality_gap':encode(upper-lower),'coefficients_sha256':S.sha(out/'coefficients.json')};S.save(out/'allocation-certificate.json',certificate)
        record.update(status='certified' if lower>allowance else 'no_certificate',exact_primal_lower=encode(lower),exact_dual_upper=encode(upper),exact_duality_gap=encode(upper-lower),primal_lower=float(lower),dual_upper=float(upper),duality_gap=float(upper-lower),reported_float_solver_slack=float(result.x[-1]),allowance=float(allowance),allocation_bound_violations=0,minimum_exact_node_slack=float(lower),dual_normalization_exact=True,relaxation_strict_certificate_impossible=upper<=0,certificate_sha256=S.sha(out/'allocation-certificate.json'),scope='Exactprimal allocation gives a globalbinary lowerbound if strictslack exceedsallowance. Exactdual only bounds bestuniformslack of this negative-edge allocationrelaxation; a negative upper bound does not disprove endpointglobaloptimality.')
        S.save(out/'allocation-result.json',record);records.append(record);S.emit('binary_allocation_bound',seed=seed,status=record['status'],primal=float(lower),dual=float(upper),gap=float(upper-lower))
    S.save(DEST/'allocation-summary.json',{'status':'complete','utc':S.utc(),'records':records,'cpu_seconds':time.perf_counter()-started})


if __name__=='__main__':main()
