"""Validate a future-only exact recurrence without changing H1ag outputs."""
from fractions import Fraction
import hashlib
import json
import time
import numpy as np
from run import ROOT,HERE,sha,save,now
from fast_mixture_thresholds import thresholds
from mixture_law_policy import crossing_thresholds


def main():
    out=ROOT/'artifacts/research/20261114/ripper/mixture_law';rows=[];timing=[]
    for eta in [Fraction(1,20),Fraction(64095,1000000),Fraction(75419,1000000)]:
        crossing_thresholds.cache_clear();start=time.perf_counter();old=crossing_thresholds(eta,2220,True);old_s=time.perf_counter()-start
        start=time.perf_counter();new=thresholds(eta,2220);new_s=time.perf_counter()-start
        assert np.array_equal(old,new)
        timing.append({'eta':str(eta),'thresholds':2221,'original_s':old_s,'recurrence_s':new_s,'observed_speedup':old_s/new_s,'array_sha256':hashlib.sha256(new.tobytes()).hexdigest()})
    cache={}
    for seed in range(32):
        path=out/f's{seed}/thresholds.json';spec=json.loads(path.read_text())
        for arm in ['mixture_exact','mixture_learned']:
            eta=Fraction(spec['rates'][arm])
            if eta not in cache:cache[eta]=thresholds(eta,2220)
            assert np.array_equal(cache[eta],spec['thresholds'][arm])
            rows.append({'seed':seed,'arm':arm,'eta':str(eta),'thresholds':2221,'source_sha256':sha(path)})
    assert np.array_equal(thresholds(Fraction(8,9),2220),np.arange(1,2222))
    report={'status':'validated','utc':now(),'scope':'Future-only implementation optimization. H1ag frozen policy, collectors, recorded thresholds and scientific outcomes were not changed or rerun.',
        'timing':timing,'recorded_table_comparisons':64,'recorded_threshold_comparisons':64*2221,'distinct_recorded_rates':len(cache),'rows':rows,
        'fallback_verified':True,'helper_sha256':sha(HERE/'fast_mixture_thresholds.py'),'original_policy_sha256':sha(HERE/'mixture_law_policy.py'),'wrapper_sha256':sha(__file__),
        'measurement_limit':'Single sequential CPU measurements per rate on the same process, while GPU inference ran on the host; no equal-load randomized timing claim. Original cache cleared before timing.'}
    save(ROOT/'artifacts/research/20261114/ripper/fast_mixture_threshold_validation.json',report)
    save(ROOT/'results/research/ripper/fast_mixture_threshold_validation.json',report);print(json.dumps({k:v for k,v in report.items() if k!='rows'}))


if __name__=='__main__':main()
