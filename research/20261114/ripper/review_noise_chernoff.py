"""Independent numeric review of the supplied-law never-equal-pair bound."""
import importlib.util
import numpy as np
from run import ROOT,HERE,sha,save,now


def main():
    path=ROOT/'research/20261114/foureyes/noise_chernoff.py'
    spec=importlib.util.spec_from_file_location('reviewed_bound',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    rng=np.random.default_rng(20261501);checks=0;maximum=0.
    for c in [0,1,3,9]:
        free=np.array([rng.random(81)<.7 for _ in range(24)]);assert np.all(free.sum(1)>=9)
        actual=mod.null_means(free,c);ii,jj=np.triu_indices(81,1);rowp=[]
        for row in free:
            m=int(row.sum());a=c/m;b=c*(c-1)/(m*(m-1))
            fi,fj=row[ii].astype(float),row[jj].astype(float)
            rowp.append((fi+fj)*a/8-fi*fj*9*b/64)
        rowp=np.array(rowp);expected=rowp.sum(0)
        assert np.allclose(actual,expected,rtol=0,atol=2e-15);checks+=len(actual)
        for pair in rng.choice(len(actual),24,replace=False):
            law=np.array([1.])
            for probability in rowp[:,pair]:law=np.convolve(law,[1-probability,probability])
            for k in range(26):
                exact=float(law[k:].sum()) if k<=24 else 0.
                upper=float(mod.chernoff_upper(k,24,np.array([actual[pair]]))[0])
                assert exact<=upper+3e-15;maximum=max(maximum,exact-upper)
                checks+=1
        threshold,mean=mod.thresholds(free,c)
        assert np.all(mod.chernoff_upper(threshold[0]+1,24,np.array([mean[0]]))<=.01/3240)
        for index in range(3240):
            t=int(threshold[index]);mu=np.array([mean[index]])
            assert mod.chernoff_upper(t+1,24,mu)[0]<=.01/3240
            if t>0:assert mod.chernoff_upper(t,24,mu)[0]>.01/3240
            checks+=1
    r={'status':'validated','utc':now(),'wrapper_sha256':sha(__file__),'reviewed_source_sha256':sha(path),
        'checks':checks,'null_mean_pair_checks':12960,'exact_poisson_binomial_tail_checks':2496,
        'minimal_threshold_checks':12960,'maximum_tail_excess_roundoff':maximum,
        'scope':'Fixed clean boards/masks and correctly supplied corruption law; independent rows; controls true-pair deletions only.','numpy':np.__version__}
    save(ROOT/'artifacts/research/20261114/ripper/noise_chernoff_review.json',r);print(r)


if __name__=='__main__':main()
