"""Benchmark exact-range float64 conflict counting versus literal int64 dot products."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3v';MODEL=ROOT/'artifacts/research/20261114/loki/h3v/models/uniform-filtered/model.npz'
def main():
    F=np.load(MODEL)['F'].astype(np.int64);x=np.random.default_rng(83299).integers(0,2,(4096,169),dtype=np.int64);timings={};values={}
    for mode in ['int64','float64_exact_integer_range']:
        a=x if mode=='int64'else x.astype(np.float64);b=F if mode=='int64'else F.astype(np.float64);times=[]
        for _ in range(3):
            start=time.perf_counter();v=(a*(a@b)).sum(1)/2;times.append(time.perf_counter()-start)
        timings[mode]=times;values[mode]=v
    assert np.array_equal(values['int64'],values['float64_exact_integer_range']);PUBLIC.mkdir(parents=True,exist_ok=True)
    p=dict(status='passed',fixture_rows=4096,cells=169,rng=83299,counts_bitwise_equal=True,reason='Every field is a sum of at most169binary terms; the double-counted total is at most169*168. All operations remain exact integers below2**53 in float64. Division by2 is exact. The audit also compares every full target solution with literal occupied-pair counts.',timings_seconds=timings,median_speedup=float(np.median(timings['int64'])/np.median(timings['float64_exact_integer_range'])),input_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()for f in [Path(__file__),MODEL]})
    (PUBLIC/'audit-count-validation.json').write_text(json.dumps(p,indent=2)+'\n');print(json.dumps(p))
if __name__=='__main__':main()
