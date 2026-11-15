"""H2bh pre-draw boundary tests against independent SciPy beta quantiles."""
import json
import numpy as np
from scipy.stats import beta
import audit as A
import study as S


def main():
    n=4096;counts=[0,1,4,8,9,20,21,40,52,64,128,2048,4095,4096];rows=[];maxerr=0
    def scipy(k,alpha):return (float(beta.ppf(alpha,k,n-k+1)) if k else 0.,float(beta.ppf(1-alpha,k+1,n-k)) if k<n else 1.)
    for alpha in [.025,.05]:
        for k in counts:
            actual=A.bounds(k,n,alpha);ref=scipy(k,alpha);err=max(abs(x-y) for x,y in zip(actual,ref));assert err<1e-10;maxerr=max(maxerr,err);rows.append({'k':k,'alpha':alpha,'bounds':actual,'maximum_error':err})
    absolutes=[]
    for failures in [0,1,8,9,52]:
        k=n-failures;lo,hi=A.bounds(k,n,.05);decision='supported' if lo>=.999 else 'falsified' if hi<.999 else 'inconclusive';assert decision==('supported' if failures==0 else 'falsified' if failures>=9 else 'inconclusive');absolutes.append({'failures':failures,'decision':decision})
    pairs=[]
    for gains,losses in [(0,0),(0,52),(4,0),(21,0),(40,0),(52,0),(64,1),(128,52),(2048,2048)]:
        gl,gu=A.bounds(gains,n,.025);ll,lu=A.bounds(losses,n,.025);sgl,sgu=scipy(gains,.025);sll,slu=scipy(losses,.025);lower=100*(gl-lu);upper=100*(gu-ll);assert abs(lower-100*(sgl-slu))<1e-8 and abs(upper-100*(sgu-sll))<1e-8
        for threshold in [.5,-.1]:
            decision='supported' if lower>=threshold else 'falsified' if upper<threshold else 'inconclusive';reference='supported' if 100*(sgl-slu)>=threshold else 'falsified' if 100*(sgu-sll)<threshold else 'inconclusive';assert decision==reference;pairs.append({'gains':gains,'losses':losses,'threshold_pp':threshold,'decision':decision})
    result={'status':'passed','n':n,'counts_checked':rows,'absolute_boundary_decisions':absolutes,'paired_decisions':pairs,'maximum_probability_error':maxerr,'new_fresh_outcomes_read':False,'audit_source_sha256':S.sha(S.HERE/'audit_confirmation96.py'),'source_sha256':S.sha(__file__),'scope':'Beforefreeze independent beta-quantile comparison at boundaries/interiorcounts, absolute99.9% classification, pairedgain0.5pp/NI-0.1pp classification, standalonealpha.05 regression bounds. No outcomes or sample expansion.'};S.save(S.OUT/'loop80_confirmation/statistics-gate.json',result);print(json.dumps({'status':'passed','bounds_checked':len(rows),'absolute_cases':len(absolutes),'paired_decisions':len(pairs),'maximum_error':maxerr}))


if __name__=='__main__':main()
