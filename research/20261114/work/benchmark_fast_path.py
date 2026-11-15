"""Like-for-like validator timing on already completed, preloaded outputs."""
import json
import time
import numpy as np
import study as S
from fast_path_validator import Validator
from larger_relaxation import feasible


def main():
    source=S.OUT/'larger_relaxation';dataset=source/'shard00.npz';outputs=source/'runs/shard00-s0/outputs.npz'
    with np.load(dataset) as x:tokens=x['tokens']
    with np.load(outputs) as x:boards=x['predictions'][0]
    validator=Validator();functions={'compiled':lambda:validator(tokens,boards),'python':lambda:np.array([feasible(t,p) for t,p in zip(tokens,boards)])}
    expected=functions['python']();assert np.array_equal(functions['compiled'](),expected);rows=[]
    for rep in range(7):
        row={'repetition':rep}
        for key in (['python','compiled'] if rep%2==0 else ['compiled','python']):
            begin=time.perf_counter();answer=functions[key]();row[key+'_seconds']=time.perf_counter()-begin;assert np.array_equal(answer,expected)
        rows.append(row)
    p=float(np.median([r['python_seconds'] for r in rows]));c=float(np.median([r['compiled_seconds'] for r in rows]))
    result={'status':'passed','utc':S.utc(),'inputs':len(tokens),'repetitions':rows,'median_python_seconds':p,'median_compiled_seconds':c,'ratio_of_medians':p/c,'method':'Same preloaded256native boards from completed H2s, conversion included, archive loading excluded.7fixed alternating repetitions after one warmup, exact feasibility parity each time. Timing only; no inference or new outcome.','dataset_sha256':S.sha(dataset),'outputs_sha256':S.sha(outputs),'script_sha256':S.sha(__file__),'validator_sha256':S.sha(S.HERE/'fast_path_validator.c')}
    S.save(S.OUT/'selective_fallback/validator-benchmark.json',result);S.emit('validator_benchmark',ratio=p/c,python_seconds=p,compiled_seconds=c)


if __name__=='__main__':main()
