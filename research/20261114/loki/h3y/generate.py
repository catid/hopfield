"""Separate privileged source generator; source inference never calls this module."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import datetime as dt
import signal
import socket
import time
import numpy as np
from common import ROOT,SRC,OUT,sha,save,read,line_enumeration,bit_enumeration

def main():
    assert socket.gethostname()=='loki';assert not(OUT/'generation.json').exists()
    p=read(SRC/'protocol.json')
    for rel,h in p['source_sha256'].items():assert sha(ROOT/rel)==h
    start=time.perf_counter();cpu=time.process_time()
    def expired(*args):raise TimeoutError('Frozen120second source-generation budget')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,120)
    try:
        lines=[(0,1),(1,-1),(1,0),(1,1)];universes={};enumeration=[];rows=[]
        for n in p['source_sizes']:
            a=bit_enumeration(n,lines);b=line_enumeration(n,lines);assert np.array_equal(a,b);universes[n]=a
            enumeration.append(dict(n=n,ordinary_boards=len(a),independent_enumerations_equal=True))
        for case in p['cases']:
            n=case['n'];family=[tuple(o)for o in p['families'][case['family_id']]];ordinary=universes[n];keep=np.ones(len(ordinary),bool)
            for r in range(n):
                for t in range(r+1,n):
                    dc=np.abs(ordinary[:,t].astype(int)-ordinary[:,r].astype(int))
                    for u,v in family:keep&=~(((t-r==u)&(dc==v))|((t-r==v)&(dc==u)))
            selected=ordinary[keep];independent=bit_enumeration(n,lines,family);assert np.array_equal(selected,independent)
            path=OUT/f'sources/{case["id"]}.npy';path.parent.mkdir(parents=True,exist_ok=True);np.save(path,selected)
            rows.append(dict(id=case['id'],n=n,source_patterns=len(selected),source_path=str(path.relative_to(ROOT)),source_sha256=sha(path),complete_source_supplied=True,independent_generation_equal=True))
        elapsed=time.perf_counter()-start;cpu_elapsed=time.process_time()-cpu;assert elapsed<120 and cpu_elapsed<120
        save(OUT/'generation.json',dict(status='complete',utc=dt.datetime.now(dt.timezone.utc).isoformat(),protocol_sha256=sha(SRC/'protocol.json'),ordinary_enumerations=enumeration,cases=rows,wall_seconds=elapsed,cpu_seconds=cpu_elapsed,no_target_enumeration=True))
        print('source generation complete',len(rows),'cases',elapsed,'seconds',flush=True)
    except BaseException as e:save(OUT/'generation-failure.json',dict(status='incomplete',error=repr(e),seconds=time.perf_counter()-start));raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__':main()
