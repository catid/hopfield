"""Exact joint unions: subset Mobius inversion and independent grouped sampling DP."""
import itertools
import math


def mobius(frequency, dose):
    width=(len(frequency)-1).bit_length(); assert len(frequency)==1<<width
    below=list(map(int,frequency))
    for bit in range(width):
        for mask in range(len(below)):
            if mask&(1<<bit): below[mask]+=below[mask^(1<<bit)]
    mass=[math.comb(count,dose) if count>=dose else 0 for count in below]
    for bit in range(width):
        for mask in range(len(mass)):
            if mask&(1<<bit): mass[mask]-=mass[mask^(1<<bit)]
    assert all(value>=0 for value in mass) and sum(mass)==math.comb(sum(frequency),dose)
    return mass


def grouped_dp(frequency, maxdose, progress=False):
    # Each group contributes j distinct boards with weight C(count,j), all
    # sharing one witness mask. No independence approximation is made.
    dp=[{} for _ in range(maxdose+1)]; dp[0][0]=1; processed=0; operations=0
    for mask,count in enumerate(frequency):
        if not count: continue
        count=int(count); choose=[math.comb(count,j) for j in range(min(count,maxdose)+1)]
        nxt=[{} for _ in range(maxdose+1)]
        for dose in range(min(maxdose,processed)+1):
            for union,weight in dp[dose].items():
                for j in range(min(count,maxdose-dose)+1):
                    key=union|mask if j else union; dest=nxt[dose+j]
                    dest[key]=dest.get(key,0)+weight*choose[j]; operations+=1
        processed+=count; dp=nxt
        if progress: print('GROUP_DP',mask,count,processed,operations,flush=True)
    assert all(sum(dp[k].values())==math.comb(processed,k) for k in range(min(processed,maxdose)+1))
    return dp,operations


def validate():
    fixtures=[[0]*8,[7]*8,[1,2,4,1,2,4,0,0],[0,7,0,7,0,7,0,7],[1,3,2,6,4,5,7,0]]
    count=0
    for masks in fixtures:
        frequency=[masks.count(i) for i in range(8)]; dp,_=grouped_dp(frequency,8)
        for dose in range(9):
            literal=[0]*8
            for chosen in itertools.combinations(masks,dose):
                union=0
                for value in chosen: union|=value
                literal[union]+=1
            assert literal==mobius(frequency,dose)==[dp[dose].get(i,0) for i in range(8)];count+=1
    return dict(fixtures=len(fixtures),dose_cases=count,all_literal_subsets_mobius_and_grouped_DP_equal=True)
