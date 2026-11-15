"""Exact joint witness-union counts for sampling subsets without replacement."""
import itertools
import math
import numpy as np

def union_mass(frequency,k):
    length=len(frequency);assert length and length&(length-1)==0;m=length.bit_length()-1;allowed=list(map(int,frequency))
    # Zeta transform: number of population boards whose witness set is a subset of S.
    for bit in range(m):
        for S in range(length):
            if S&(1<<bit):allowed[S]+=allowed[S^(1<<bit)]
    below=[math.comb(n,k)if n>=k else 0 for n in allowed];exact=below.copy()
    # Integer Mobius inversion: draws with union exactly S.
    for bit in range(m):
        for S in range(length):
            if S&(1<<bit):exact[S]-=exact[S^(1<<bit)]
    assert all(x>=0 for x in exact)and sum(exact)==math.comb(sum(frequency),k)
    reconstructed=exact.copy()
    for bit in range(m):
        for S in range(length):
            if S&(1<<bit):reconstructed[S]+=reconstructed[S^(1<<bit)]
    assert reconstructed==below
    return exact

def validate():
    rng=np.random.default_rng(2026113110);cases=[]
    fixtures=[(0,[0]*8),(1,[0]*8),(3,[7]*8),(3,[1,2,4,1,2,4,0,0]),(3,[0,7,0,7,0,7,0,7])]
    fixtures += [(m,rng.integers(0,1<<m,8).tolist())for m in range(1,6)for _ in range(4)]
    for case,(m,masks)in enumerate(fixtures):
        freq=np.bincount(masks,minlength=1<<m).tolist()
        for k in range(9):
            literal=[0]*(1<<m)
            for selected in itertools.combinations(range(8),k):
                union=0
                for i in selected:union|=masks[i]
                literal[union]+=1
            computed=union_mass(freq,k);assert computed==literal;cases.append(dict(case=case,classes=m,population=8,draw=k,total_subsets=sum(computed),every_union_count_identical=True))
    return dict(status='passed',fixtures=len(fixtures),cases=cases,all_exhaustive_small_subset_counts_identical=True,no_marginal_independence_assumption=True)
