"""Independent public-only exact novelty and previous-margin audit for H2ad."""
from fractions import Fraction as Q
import hashlib
import json
import numpy as np


def counts(t,b,mu):
    p=np.where(t==1,b,t>=2).astype(np.int8);height,width=p.shape;values=[]
    for dr,dc in mu:
        dr,dc=int(dr),int(dc);r0,r1=max(0,-dr),min(height,height-dr);c0,c1=max(0,-dc),min(width,width-dc)
        cells=2*p[r0:r1,c0:c1]+p[r0+dr:r1+dr,c0+dc:c1+dc];values.extend(map(int,np.bincount(cells.ravel(),minlength=4)))
    values.extend([-int(((t==1)&(p==0)).sum()),-int(((t==1)&(p==1)).sum())]);return values


def feature(t,b,a,mu,ao):
    p=np.where(t==1,b,t>=2).astype(np.int8);height,width=p.shape;values=list(map(Q,counts(t,b,mu)))
    for k,(dr,dc) in enumerate(ao):
        total=Q(0);dr,dc=int(dr),int(dc)
        for r in range(height):
            for c in range(width):
                rr,cc=r+dr,c+dc
                if 0<=rr<height and 0<=cc<width and p[r,c] and p[rr,cc]:total-=Q.from_float(float(a[k,r,c]))
        values.append(total)
    return values


def normalized(t,y,b,a,mu,ao,h):return tuple((z-v)/int(h) for z,v in zip(feature(t,b,a,mu,ao),feature(t,y,a,mu,ao)))


def verify_seed(directory,tokens,labels,attention,seed,mu,ao):
    receipt=json.loads((directory/'novelty-audit.json').read_text())['records'][seed]
    with np.load(directory/f'negatives-seed{seed}.npz',allow_pickle=False) as z:indices=z['indices'];boards=z['boards'];hamming=z['hamming']
    with np.load(directory/f'previous-negatives-seed{seed}.npz',allow_pickle=False) as z:oi=z['indices'];ob=z['boards'];oh=z['hamming']
    theta=np.load(directory/f'previous-theta-seed{seed}.npy',allow_pickle=False);new=[normalized(tokens[i],labels[i],b,attention[i],mu,ao,h) for i,b,h in zip(indices,boards,hamming)]
    # Every source single-cell flip has integer pair-table and bias coefficients.
    # Each observed new row has a fractional coefficient there, excluding all flips.
    assert all(any(v.denominator>1 for v in row[:18]) for row in new)
    prefix={row[:18] for row in new};prior=set();compared=0
    for i,(t,y,a) in enumerate(zip(tokens,labels,attention)):
        canonical=counts(t,y,mu)
        for b,h in [(np.zeros_like(y),int(((t==1)&y).sum())),(np.ones_like(y),int(((t==1)&~y).sum()))]:
            key=tuple(Q(v-z,h) for v,z in zip(counts(t,b,mu),canonical))
            if key in prefix:prior.add(normalized(t,y,b,a,mu,ao,h));compared+=1
    for i,b,h in zip(oi,ob,oh):
        key=tuple(Q(v-z,int(h)) for v,z in zip(counts(tokens[i],b,mu),counts(tokens[i],labels[i],mu)))
        if key in prefix:prior.add(normalized(tokens[i],labels[i],b,attention[i],mu,ao,h));compared+=1
    oldboard={(int(i),b.tobytes()) for i,b in zip(oi,ob)};seen=set(prior);novelboard=novelrow=violation=0
    for j,(i,b,row) in enumerate(zip(indices,boards,new)):
        r=receipt['mapping'][j];key=(int(i),b.tobytes());bn=key not in oldboard;rn=row not in seen;oldboard.add(key);seen.add(row);margin=sum(v*Q.from_float(float(w)) for v,w in zip(row,theta));violates=margin<1
        assert bn==r['new_board_vs_all_previous'] and rn==r['new_exact_normalized_constraint_vs_all_previous'] and violates==r['violates_previous_unit_margin_exact']
        assert margin==Q(int(r['previous_exact_margin_numerator']),int(r['previous_exact_margin_denominator']))
        assert hashlib.sha256(json.dumps([[v.numerator,v.denominator] for v in row],separators=(',',':')).encode()).hexdigest()==r['exact_constraint_sha256']
        novelboard+=bn;novelrow+=rn;violation+=violates
    assert novelboard==receipt['genuinely_new_boards'] and novelrow==receipt['genuinely_new_exact_constraint_rows'] and violation==receipt['previous_unit_margin_violations']
    return {'seed':seed,'new_boards':novelboard,'new_exact_constraints':novelrow,'previous_unit_margin_violations':violation,'exact_prior_comparisons':compared,'single_flip_rows_excluded_by_fractional_pair_or_bias_coordinate':564329,'scope':'Independent integer pair histograms and exact rational directed-link sums. All prior single flips excluded algebraically by fractional pair/bias coordinates; every prior endpoint/all-open and mined row checked by exact first18-coordinate identity, with complete exact row comparison for matches.'}
