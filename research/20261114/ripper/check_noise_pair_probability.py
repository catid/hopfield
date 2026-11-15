"""Exact rational enumeration of root's pair-probability noise formula."""
from fractions import Fraction as F
from itertools import combinations
from math import comb
from run import ROOT,save,sha,now


def main():
    checks=[]
    for m in range(9):
        for c in range(m+1):
            a=F(c,m) if m else F(0)
            b=F(c*(c-1),m*(m-1)) if m>=2 else F(0)
            kinds=[('neither',m,m+1)]
            if m>=1:kinds.append(('one',0,m))
            if m>=2:kinds.append(('both',0,1))
            for kind,i,j in kinds:
                for equal in (False,True):
                    old_i,old_j=0,0 if equal else 1
                    probability=F(0)
                    for subset in combinations(range(m),c):
                        selected=set(subset)
                        vi=[v for v in range(9) if v!=old_i] if i in selected else [old_i]
                        vj=[v for v in range(9) if v!=old_j] if j in selected else [old_j]
                        probability+=F(sum(x==y for x in vi for y in vj),len(vi)*len(vj)*comb(m,c))
                    if kind=='both':formula=1-2*a+F(9,8)*b if equal else a/4-F(9,64)*b
                    elif kind=='one':formula=1-a if equal else a/8
                    else:formula=F(int(equal))
                    assert probability==formula,(m,c,kind,equal,probability,formula)
                    checks.append({'m':m,'c':c,'free_status':kind,'initially_equal':equal,
                                   'probability_numerator':probability.numerator,'probability_denominator':probability.denominator})
    result={'status':'validated_exact_rational','utc':now(),'wrapper_sha256':sha(__file__),
        'cases':len(checks),'m_range_inclusive':[0,8],'rows':checks,
        'assumptions':'Exactly c of m free cells sampled uniformly; each selected digit independently replaced uniformly among its other8 values.'}
    path=ROOT/'artifacts/research/20261114/ripper/noise_pair_formula_check.json';save(path,result)
    print({'cases':len(checks),'status':result['status'],'path':str(path)})


if __name__=='__main__':main()
