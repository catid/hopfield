"""Independent literal source certificate; cached independent universe is hash-bound."""
import itertools
import math
import numpy as np
from common import bits

def verify(columns,n,universe,independent,lines,generators,incidence,positive,checked,minima,minimum):
    assert np.array_equal(universe,independent)
    counts={(r,c):0 for r in range(n)for c in range(1-n,n)if r or c>0}
    for cols in columns:
        for r in range(n):
            for t in range(r+1,n):counts[t-r,int(cols[t])-int(cols[r])]+=1
    normals=[]
    for a in range(3):
        for b in range(-2,3):
            if (a==0 and b<=0)or math.gcd(a,abs(b))!=1:continue
            values=[count for(r,c),count in counts.items()if a*r==-b*c]
            if len(values)>=3 and all(v==0 for v in values):normals.append((a,b))
    assert normals==lines
    zeros={o for o,v in counts.items()if v==0 and not any(a*o[0]+b*o[1]==0 for a,b in normals)}
    candidates=sorted({(min(abs(r),abs(c)),max(abs(r),abs(c)))for r,c in zeros});assert candidates==generators
    source_keys={tuple(map(int,row))for row in columns};flags=np.array([tuple(map(int,row))in source_keys for row in independent]);assert np.array_equal(flags,positive)
    assert int(flags.sum())==len(columns)==len(source_keys)
    rebuilt=np.zeros(incidence.shape,bool)
    for k,row in enumerate(independent.astype(int)):
        pairsets={(min(t-r,abs(int(row[t]-row[r]))),max(t-r,abs(int(row[t]-row[r]))))for r,t in itertools.combinations(range(n),2)}
        rebuilt[k]=[g in pairsets for g in candidates]
    assert np.array_equal(rebuilt,incidence)and not rebuilt[flags].any()
    neg=np.flatnonzero(~flags);allbits=(1<<len(neg))-1;coverbits=[bits(rebuilt[neg,k])for k in range(len(candidates))]
    expected_choices=[list(c)for size in range(minimum+1)for c in itertools.combinations(range(len(candidates)),size)]
    assert len(expected_choices)==len(checked)
    found=[]
    for expected,rec in zip(expected_choices,checked):
        assert rec['indices']==expected;union=0
        for k in expected:union|=coverbits[k]
        missing=allbits^union;witness=int(neg[(missing&-missing).bit_length()-1])if missing else None
        assert rec['rejected_negatives']==union.bit_count()and rec['first_uncovered_universe_index']==witness
        assert rec['first_uncovered_columns']==(independent[witness].astype(int).tolist()if witness is not None else None)
        assert rec['covers_all_negatives']==(missing==0)
        if not missing:found.append(expected);assert len(expected)==minimum
    assert found==minima
    return dict(status='passed',independent_universe_equal=True,literal_source_counts_and_lines_equal=True,literal_pair_incidence_equal=True,
                every_lower_and_equal_cardinality_subset_checked=len(checked),all_noncover_witnesses_verified=True,all_minima=found)
