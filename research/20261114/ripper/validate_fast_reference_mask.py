"""Parity and timing gate only; does not modify any frozen experiment or audit."""
import json
import time
import numpy as np
from run import ROOT,HERE,sha,save,now
from audit_scaled_noise_acquisition import reference_mask as original
from fast_reference_mask import reference_mask as fast


def main():
    rng=np.random.default_rng(20333000);cases=[]
    for index in range(512):
        joint=rng.integers(0,500 if index%2 else 15,(81,81),dtype=np.int32);joint=(joint+joint.T)//2
        excluded=rng.random((81,81))<(index%10)/10;excluded=excluded|excluded.T
        if index%32==0:joint[:]=0
        if index%64==0:excluded[:]=True
        order=rng.permutation(81);joint.setflags(write=False);excluded.setflags(write=False);cases.append((joint,excluded,order))
    results={};timings={}
    for name,fn in [('reference',original),('fast',fast)]:
        t=time.perf_counter();results[name]=[fn('exact_long',j,e,o) for j,e,o in cases];timings[name]=time.perf_counter()-t
    for x,y in zip(results['reference'],results['fast']):assert np.array_equal(x,y)
    r={'status':'validated','utc':now(),'wrapper_sha256':sha(__file__),'helper_sha256':sha(HERE/'fast_reference_mask.py'),
       'reference_sha256':sha(HERE/'audit_scaled_noise_acquisition.py'),'cases':len(cases),'timings_s':timings,
       'speedup':timings['reference']/timings['fast'],'scope':'Independent floating-operation parity; no experiment runner or audit was replaced.','numpy':np.__version__}
    save(ROOT/'artifacts/research/20261114/ripper/fast_mask_validation.json',r);print(json.dumps(r))


if __name__=='__main__':main()
