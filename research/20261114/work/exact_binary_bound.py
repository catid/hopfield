"""Exact dyadic QUBO and half-allocation certificate for one preselected graph."""
import argparse
from fractions import Fraction as Q
import json
import math
import time
import numpy as np
import study as S

DEST=S.OUT/'global_bound'
PP=S.HERE/'global-bound-protocol.json'


def rational(x):return Q.from_float(float(x))


def derive(tokens,tables,bias,kappa,attention,mu_offsets,aoffs):
    h,w=tokens.shape;free=np.flatnonzero(tokens.ravel()==1);index={int(q):i for i,q in enumerate(free)};n=len(free);zero=Q(0);pieces={key:[zero]*n for key in ['bias','mu_pair','self_link','clamped_link']};edges={};absolute=zero
    table=[[[rational(x) for x in row] for row in t] for t in tables];b=list(map(rational,bias));k=list(map(rational,kappa));att=[[rational(x) for x in a.ravel()] for a in attention]
    pieces['bias']=[b[0]-b[1] for _ in free];absolute+=n*max(map(abs,b))
    def edge(i,j,value):
        key=tuple(sorted((i,j)));edges[key]=edges.get(key,zero)+value
    for z,(dr,dc) in enumerate(mu_offsets):
        tab=table[z];bound=max(abs(v) for row in tab for v in row)
        for r in range(h):
            for c in range(w):
                rr,cc=r+dr,c+dc
                if not(0<=rr<h and 0<=cc<w):continue
                i=r*w+c;j=rr*w+cc;fi=i in index;fj=j in index;ci=int(tokens[r,c]>=2);cj=int(tokens[rr,cc]>=2);absolute+=bound
                if fi and fj:
                    ii,jj=index[i],index[j];pieces['mu_pair'][ii]+=tab[1][0]-tab[0][0];pieces['mu_pair'][jj]+=tab[0][1]-tab[0][0];edge(ii,jj,tab[1][1]-tab[1][0]-tab[0][1]+tab[0][0])
                elif fi:pieces['mu_pair'][index[i]]+=tab[1][cj]-tab[0][cj]
                elif fj:pieces['mu_pair'][index[j]]+=tab[ci][1]-tab[ci][0]
    for z,(dr,dc) in enumerate(aoffs):
        for r in range(h):
            for c in range(w):
                rr,cc=r+dr,c+dc
                if not(0<=rr<h and 0<=cc<w):continue
                i=r*w+c;j=rr*w+cc;value=-k[z]*att[z][i];absolute+=abs(value)
                if not tokens[r,c] or not tokens[rr,cc]:continue
                fi=i in index;fj=j in index
                if fi and fj:
                    if i==j:pieces['self_link'][index[i]]+=value
                    else:edge(index[i],index[j],value)
                elif fi:pieces['clamped_link'][index[i]]+=value
                elif fj:pieces['clamped_link'][index[j]]+=value
    unary=[sum((v[i] for v in pieces.values()),zero) for i in range(n)];edges={k:v for k,v in edges.items() if v}
    return free,unary,edges,pieces,absolute


def encode(q):return {'numerator':str(q.numerator),'denominator':str(q.denominator)}


def run():
    started=time.perf_counter();DEST.mkdir(parents=True,exist_ok=True);p=json.loads(PP.read_text());torch,_,_=S.imports();records=[]
    for path,h in p['source_sha256'].items():assert S.sha(S.ROOT/path)==h
    for seed in range(3):
        out=DEST/f'seed{seed}';out.mkdir(exist_ok=True)
        if (out/'half-result.json').exists():raise FileExistsError(out)
        source=p['graph_states'][str(seed)];assert S.sha(S.ROOT/source['path'])==source['sha256'];assert S.sha(S.ROOT/p['models'][str(seed)]['path'])==p['models'][str(seed)]['sha256']
        with np.load(S.ROOT/source['path']) as state:tokens=state['tokens'];attention=state['couplings']
        model=S.model_for(seed,90,'original','cpu',torch.float64);tables=model.T.detach().numpy();bias=model.b.detach().numpy();kappa=model.kappa.detach().numpy();mu_offsets=list(model.mu_offsets);aoffs=list(model.aoffs)
        # Every original scalar was float32; float64 promotes it exactly.
        assert np.array_equal(tables,tables.astype(np.float32).astype(np.float64)) and attention.dtype==np.float32
        free,unary,edges,pieces,absolute=derive(tokens,tables,bias,kappa,attention,mu_offsets,aoffs)
        denominator=max([q.denominator for q in unary]+[q.denominator for q in edges.values()]+[q.denominator for values in pieces.values() for q in values]);assert denominator&(denominator-1)==0
        hu=[int(q*denominator) for q in unary];pairs=sorted(edges);ju=[int(edges[e]*denominator) for e in pairs];half_twice=[2*x for x in hu]
        for (i,j),value in zip(pairs,ju):
            if value<0:half_twice[i]+=value;half_twice[j]+=value
        half_min=Q(min(half_twice),2*denominator)
        # Deliberately loose evaluation allowance. Binary entropy is constant;
        # 256*H*W+1000 exceeds multiply/add count in source energy per board.
        # gamma_m=sum-product bound, two evaluations + reconstruction allowance.
        operations=256*tokens.size+1000;unit=Q(1,2**53);gamma=operations*unit/(1-operations*unit);allowance=4*gamma*(absolute+Q(len(free),10**11))+Q(1,10**8)
        tensor=torch.tensor(tokens.astype(np.int64))[None];a=torch.tensor(attention,dtype=torch.float64)[None];zero=torch.zeros((1,*tokens.shape),dtype=torch.float64)
        with torch.no_grad():
            base=model.energy(tensor,zero,a)[0];h_native=model.flip_delta(tensor,zero,a)[0].numpy().ravel()[free]
            h_error=float(np.max(np.abs(h_native-np.array([float(x) for x in unary]))));assert h_error<float(allowance)
            # Nine-color discrete probes recover every local pair coefficient:
            # at most one neighbor of each color is within a3x3window.
            colors=(np.indices(tokens.shape)[0]%3)*3+np.indices(tokens.shape)[1]%3;colored={}
            for color in range(9):
                board=torch.tensor(((tokens==1)&(colors==color)).astype(np.float64))[None];colored[color]=model.flip_delta(tensor,board,a)[0].numpy().ravel()[free]
            errors=[]
            for (i,j),value in edges.items():
                color=int(colors.ravel()[free[j]]);assert colors.ravel()[free[i]]!=color;errors.append(abs(float(value)-(colored[color][i]-h_native[i])))
            pair_error=max(errors,default=0.);assert pair_error<float(allowance)
            rng=np.random.default_rng(np.random.SeedSequence([p['probe_seed'],seed]));bits=rng.integers(0,2,(p['probe_boards'],len(free)),dtype=np.uint8);boards=np.zeros((len(bits),tokens.size),np.float64);boards[:,free]=bits;native=model.energy(tensor.expand(len(bits),-1,-1),torch.tensor(boards.reshape(-1,*tokens.shape)),a.expand(len(bits),-1,-1,-1))-base
            exact=[]
            for b in bits:exact.append(float(Q(sum(x for x,on in zip(hu,b) if on)+sum(v for (i,j),v in zip(pairs,ju) if b[i] and b[j]),denominator)))
            energy_error=float(np.max(np.abs(native.numpy()-exact)));assert energy_error<float(allowance)
        np.savez_compressed(out/'constants.npz',tokens=tokens,attention=attention,T=tables.astype(np.float32),b=bias.astype(np.float32),kappa=kappa.astype(np.float32),mu_offsets=np.array(mu_offsets),aoffs=np.array(aoffs))
        coefficients={'exact_constant_scope':'Float32 archived readout and learned tensors interpreted as exactdyadic constants; binary entropy is state-independent and cancels.','free_flat_indices':free.tolist(),'common_denominator':str(denominator),'unary_numerators':list(map(str,hu)),'edge_nodes':[list(e) for e in pairs],'edge_numerators':list(map(str,ju)),'unary_contribution_numerators':{k:[str(int(x*denominator)) for x in values] for k,values in pieces.items()},'roundoff_allowance':encode(allowance),'half_slack_twice_numerators':list(map(str,half_twice)),'constants_sha256':S.sha(out/'constants.npz')};S.save(out/'coefficients.json',coefficients)
        record={'seed':seed,'status':'certified' if half_min>allowance else 'no_certificate','exact_half_minimum_slack':encode(half_min),'minimum_slack_float':float(half_min),'nonpositive_half_slack_nodes':sum(x<=0 for x in half_twice),'nodes':len(free),'edges':len(pairs),'negative_edges':sum(x<0 for x in ju),'source_float64_allowance':float(allowance),'allowance_exact':encode(allowance),'allowance_method':{'operation_upper_bound_per_board':operations,'unit_roundoff':'2^-53','absolute_term_upper_bound':encode(absolute),'formula':'4*gamma_m*(absolute_term_bound+nfree/10^11)+10^-8, gamma_m=m*u/(1-m*u). Exactrational coefficient/certificate proof has no rounding. This conservative allowance is only for the float64source checker, not a float32 summation guarantee.'},'source_parity':{'all_unary_checks':len(free),'all_pair_checks':len(pairs),'random_binary_boards':p['probe_boards'],'maximum_unary_error':h_error,'maximum_pair_error':pair_error,'maximum_energy_difference_error':energy_error},'protocol_sha256':S.sha(PP),'script_sha256':S.sha(__file__),'constants_sha256':S.sha(out/'constants.npz'),'coefficients_sha256':S.sha(out/'coefficients.json'),'checkpoint_sha256':p['models'][str(seed)]['sha256'],'graph_state_sha256':source['sha256'],'scope':'One preselectedgraph, fixed retainedreadout. Half allocation failure says only no certificate; no globaloptimality conclusion.'}
        S.save(out/'half-result.json',record);records.append(record);S.emit('binary_half_bound',seed=seed,status=record['status'],slack=record['minimum_slack_float'],allowance=float(allowance))
    S.save(DEST/'half-summary.json',{'status':'complete','utc':S.utc(),'records':records,'cpu_seconds':time.perf_counter()-started,'protocol_sha256':S.sha(PP)});S.emit('half_bound_complete',seconds=time.perf_counter()-started)


if __name__=='__main__':run()
