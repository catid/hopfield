"""Standalone exact verifier: public files + Python/NumPy only; no Torch/SciPy/upstream."""
import argparse
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def q(x):return Q.from_float(float(x))
def decode(x):return Q(int(x['numerator']),int(x['denominator']))


def independently_reconstruct(z):
    tok=z['tokens'];height,width=tok.shape;free=np.flatnonzero(tok.ravel()==1);node={int(k):i for i,k in enumerate(free)};tables=[[[q(v) for v in row] for row in t] for t in z['T']];bias=list(map(q,z['b']));kappa=list(map(q,z['kappa']));attention=[[q(v) for v in a.ravel()] for a in z['attention']];muoffs=list(map(tuple,z['mu_offsets'].tolist()));aoffs=list(map(tuple,z['aoffs'].tolist()));offset_index={d:i for i,d in enumerate(aoffs)};zero=Q(0);parts={k:[] for k in ['bias','mu_pair','self_link','clamped_link']};pairs={};absolute=len(free)*max(map(abs,bias))
    # Independent per-node calculation from direct table changes at an off cell.
    for flat in free:
        r,c=divmod(int(flat),width);mu=zero;clamped=zero
        for table,(dr,dc) in zip(tables,muoffs):
            rr,cc=r+dr,c+dc
            if 0<=rr<height and 0<=cc<width:
                fixed=int(tok[rr,cc]>=2);mu+=table[1][fixed]-table[0][fixed]
            rr,cc=r-dr,c-dc
            if 0<=rr<height and 0<=cc<width:
                fixed=int(tok[rr,cc]>=2);mu+=table[fixed][1]-table[fixed][0]
        for d,(dr,dc) in enumerate(aoffs):
            if (dr,dc)==(0,0):continue
            rr,cc=r+dr,c+dc
            if 0<=rr<height and 0<=cc<width and tok[rr,cc]>=2:clamped-=kappa[d]*attention[d][flat]
            rr,cc=r-dr,c-dc
            if 0<=rr<height and 0<=cc<width and tok[rr,cc]>=2:clamped-=kappa[d]*attention[d][rr*width+cc]
        self_index=offset_index[(0,0)];parts['bias'].append(bias[0]-bias[1]);parts['mu_pair'].append(mu);parts['self_link'].append(-kappa[self_index]*attention[self_index][flat]);parts['clamped_link'].append(clamped)
        for table,(dr,dc) in zip(tables,muoffs):
            rr,cc=r+dr,c+dc
            if not(0<=rr<height and 0<=cc<width) or tok[rr,cc]!=1:continue
            other=rr*width+cc;value=table[1][1]+table[0][0]-table[1][0]-table[0][1]
            if (dr,dc) in offset_index:
                d=offset_index[(dr,dc)];back=offset_index[(-dr,-dc)];value-=kappa[d]*attention[d][flat]+kappa[back]*attention[back][other]
            if value:pairs[tuple(sorted((node[int(flat)],node[other])))]=value
    # Reproduce the declared conservative float64comparison allowance exactly.
    for table,(dr,dc) in zip(tables,muoffs):absolute+=(height-abs(dr))*(width-abs(dc))*max(abs(v) for row in table for v in row)
    for d,(dr,dc) in enumerate(aoffs):
        for r in range(height):
            for c in range(width):
                if 0<=r+dr<height and 0<=c+dc<width:absolute+=abs(kappa[d]*attention[d][r*width+c])
    operations=256*tok.size+1000;unit=Q(1,2**53);gamma=operations*unit/(1-operations*unit);allowance=4*gamma*(absolute+Q(len(free),10**11))+Q(1,10**8)
    return free,parts,pairs,allowance


def verify(directory):
    started=time.perf_counter();summaries=[]
    for seed in range(3):
        folder=directory/f'seed{seed}';coeff=json.loads((folder/'coefficients.json').read_text());cert=json.loads((folder/'allocation-certificate.json').read_text());half=json.loads((folder/'half-result.json').read_text());result=json.loads((folder/'allocation-result.json').read_text());assert sha(folder/'constants.npz')==coeff['constants_sha256'];assert sha(folder/'coefficients.json')==cert['coefficients_sha256']==half['coefficients_sha256']==result['coefficients_sha256'];assert sha(folder/'allocation-certificate.json')==result['certificate_sha256']
        with np.load(folder/'constants.npz',allow_pickle=False) as z:free,parts,edges,allowance=independently_reconstruct(z)
        denominator=int(coeff['common_denominator']);assert denominator>0 and denominator&(denominator-1)==0;assert free.tolist()==coeff['free_flat_indices']
        for key,values in parts.items():assert values==[Q(int(x),denominator) for x in coeff['unary_contribution_numerators'][key]],key
        h=[sum((values[i] for values in parts.values()),Q(0)) for i in range(len(free))];assert h==[Q(int(x),denominator) for x in coeff['unary_numerators']]
        stored={tuple(edge):Q(int(value),denominator) for edge,value in zip(coeff['edge_nodes'],coeff['edge_numerators'])};assert stored==edges;assert allowance==decode(coeff['roundoff_allowance'])==decode(half['allowance_exact'])
        negative=[(edge,-value) for edge,value in sorted(edges.items()) if value<0];half_slack=h.copy()
        for (i,j),weight in negative:half_slack[i]-=weight/2;half_slack[j]-=weight/2
        assert half_slack==[Q(int(x),2*denominator) for x in coeff['half_slack_twice_numerators']];assert min(half_slack)==decode(half['exact_half_minimum_slack'])
        assert cert['negative_edge_order']==[list(edge) for edge,_ in negative];allocation=list(map(decode,cert['edge_allocation_to_first_vertex']));assert len(allocation)==len(negative);loads=[Q(0)]*len(h)
        for ((i,j),weight),x in zip(negative,allocation):assert 0<=x<=weight;loads[i]+=x;loads[j]+=weight-x
        slack=[value-load for value,load in zip(h,loads)];assert slack==list(map(decode,cert['node_slacks']));lower=min(slack);assert lower==decode(cert['exact_primal_lower'])==decode(result['exact_primal_lower']);assert lower>allowance>0
        dual=list(map(decode,cert['normalized_nonnegative_dual']));assert len(dual)==len(h) and sum(dual,Q(0))==1 and all(x>=0 for x in dual)
        upper=sum((x*y for x,y in zip(h,dual)),Q(0))-sum((weight*min(dual[i],dual[j]) for (i,j),weight in negative),Q(0));assert upper==decode(cert['exact_dual_upper'])==decode(result['exact_dual_upper']);assert upper-lower==decode(cert['exact_duality_gap']) and upper>=lower
        summaries.append({'seed':seed,'status':'exact_certificate_passed','free_binary_cells':len(h),'pair_coefficients':len(edges),'negative_edges':len(negative),'half_allocation_minimum_slack':float(min(half_slack)),'exact_minimum_primal_slack':float(lower),'exact_dual_upper':float(upper),'exact_duality_gap':float(upper-lower),'conservative_float64_allowance':float(allowance),'strict_slack_over_allowance_ratio':float(lower/allowance),'conclusion':'With storedreadout and clampedwalls/endpoints, endpoint-only is the unique globalminimum over allfree binaryanswers. For anynonzero freeboard, E(board)-E(endpoint)>=minimum_primal_slack*number_selected_free_cells>0.'})
    return {'status':'passed','cpu_seconds':time.perf_counter()-started,'records':summaries,'scope':'Exactrational check of publicrawconstants, independentper-node coefficient reconstruction, allallocation bounds/slacks and normalizeddual. No Torch/SciPy/upstream/artifacts required. Does not claim a continuous minimum or generalize to anothergraph/readout.','verifier_sha256':sha(Path(__file__))}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',nargs='?',type=Path,default=Path(__file__).resolve().parents[3]/'results/research/work/global-bound');args=parser.parse_args();print(json.dumps(verify(args.directory),indent=2))
