"""Independent SciPy decision boundaries, before H2bk outcome generation."""
import json
from scipy.stats import beta
import study as S
import audit as A
from fallback_confirmation_statistics import absolute,noninferiority


def main():
    n=4096;rows=[];maxerror=0.
    def reference(k,alpha):return (float(beta.ppf(alpha,k,n-k+1)) if k else 0.,float(beta.ppf(1-alpha,k+1,n-k)) if k<n else 1.)
    for alpha in [.025,.05]:
        for k in [0,1,4,8,9,20,21,40,52,64,128,2048,4095,4096]:
            got=A.bounds(k,n,alpha);ref=reference(k,alpha);error=max(abs(x-y) for x,y in zip(got,ref));assert error<1e-10;maxerror=max(maxerror,error);rows.append({'k':k,'alpha':alpha,'maximum_error':error})
    absolute_rows=[]
    for fails in [0,1,4,8,9,52]:
        row=absolute(n-fails);lo,hi=reference(n-fails,.05);decision='supported' if lo>=.999 else 'falsified' if hi<.999 else 'inconclusive';assert row['decision']==decision;absolute_rows.append(row)
    paired=[]
    for g,l in [(0,0),(0,1),(0,4),(0,52),(4,0),(21,0),(40,0),(52,0),(64,1),(128,52),(2048,2048)]:
        row=noninferiority(g,l);gl,gu=reference(g,.025);ll,lu=reference(l,.025);lower=100*(gl-lu);upper=100*(gu-ll);assert abs(lower-row['separately_one_sided95_net_lower_pp'])<1e-8 and abs(upper-row['separately_one_sided95_net_upper_pp'])<1e-8;assert row['decision']==('supported' if lower>=-.1 else 'falsified' if upper<-.1 else 'inconclusive');paired.append(row)
    result={'status':'passed','bound_cases':rows,'absolute_cases':absolute_rows,'paired_cases':paired,
        'maximum_probability_error':maxerror,'fresh_outcomes_read':False,'source_sha256':{n:S.sha(S.HERE/n) for n in ['fallback_confirmation_statistics.py','preflight_fallback_confirmation_statistics.py','audit_fallback_confirmation.py']}}
    S.save(S.OUT/'fallback_confirmation/statistics-gate.json',result);print(json.dumps({'status':'passed','maximum_probability_error':maxerror}))


if __name__=='__main__':main()
