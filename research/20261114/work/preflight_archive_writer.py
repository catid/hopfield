"""Fixed CPU writer parity/ownership/backpressure/error gates; no timings endpoint."""
import json
from pathlib import Path
import time
import numpy as np
import study as S
from bounded_archive_writer import ArchiveWriter

D=S.OUT/'artifact_overlap_preflight'


def main():
    assert not (D/'synthetic-gate.json').exists();D.mkdir(exist_ok=True);started=time.perf_counter();rng=np.random.default_rng(2026111537);base=rng.normal(size=(17,13)).astype(np.float64);payloads=[{'a':np.arange(31,dtype=np.int16),'mask':np.arange(31)%3==0},{'a':base,'f':np.asfortranarray(base),'slice':base[::2,::3]},{'z':(base+1j*base).astype(np.complex64),'empty':np.zeros((0,0,180,180),bool)},{'trace_utf8':np.frombuffer(json.dumps({'round':0,'chosen':[2],'delta':-1e-9}).encode(),np.uint8),'indices':np.array([0,17,31],np.int64)}];results={}
    for name,asynchronous in [('serialized',False),('async',True)]:
        writer=ArchiveWriter(asynchronous)
        for i in range(24):writer.submit([(D/name/f'case{i:02d}.npz',payloads[i%len(payloads)])])
        receipts=writer.close();assert len(receipts)==24 and writer.closed and writer.metrics['maximum_pending']<=2;results[name]=writer.metrics
    for i in range(24):
        old=D/'serialized'/f'case{i:02d}.npz';new=D/'async'/f'case{i:02d}.npz';assert S.sha(old)==S.sha(new)
        with np.load(old) as x,np.load(new) as y:assert x.files==y.files and all(np.array_equal(x[k],y[k]) for k in x.files)
    # Mutating the producer buffer after enqueue must not change its written copy.
    array=np.arange(1000,dtype=np.float32);expected=array.copy();writer=ArchiveWriter(True);writer.submit([(D/'ownership.npz',{'array':array})]);array[:]=-7;writer.close()
    with np.load(D/'ownership.npz') as z:assert np.array_equal(z['array'],expected)
    failures=[]
    for asynchronous in [False,True]:
        name='async' if asynchronous else 'serialized';writer=ArchiveWriter(asynchronous,fail_name='fail.npz');caught=False
        try:
            writer.submit([(D/'error'/name/'ok.npz',{'array':expected}),(D/'error'/name/'fail.npz',{'array':expected})]);writer.close()
        except RuntimeError as e:caught='Injected archive failure' in str(e)
        finally:
            if not writer.closed:
                try:writer.close()
                except RuntimeError:pass
        assert caught and writer.closed and not (D/'error'/name/'fail.npz').exists();failures.append({'mode':name,'injected_error_propagated':True,'writer_joined':True})
    assert time.perf_counter()-started<120;S.save(D/'synthetic-gate.json',{'status':'passed','byte_identical_files':24,'varied_layout_payloads':len(payloads),'ownership_copy_verified':True,'max_pending':2,'metrics':results,'error_cases':failures,'seconds':time.perf_counter()-started,'source_sha256':{str((S.HERE/n).relative_to(S.ROOT)):S.sha(S.HERE/n) for n in ['bounded_archive_writer.py','preflight_archive_writer.py','fast_npz.py']},'scope':'Deterministic lossless byte/array parity, bounded producer queue, owned buffers and synchronous/asynchronous failure propagation. Gate runtime only, not speed endpoint.'})


if __name__=='__main__':main()
