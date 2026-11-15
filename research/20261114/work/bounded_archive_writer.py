"""Lossless serialized/bounded asynchronous archives with explicit completion."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import resource
import threading
import time
import numpy as np
import fast_npz as Z
import study as S


class ArchiveWriter:
    def __init__(self,asynchronous,max_pending=2,memory_gib=8,fail_name=None):
        self.asynchronous=asynchronous;self.max_pending=max_pending;self.memory_limit_kib=memory_gib*1024**2;self.fail_name=fail_name;self.pool=ThreadPoolExecutor(max_workers=2,thread_name_prefix='lossless-archive') if asynchronous else None;self.pending=deque();self.results=[];self.closed=False;self.lock=threading.Lock();self.metrics={'ownership_copy_seconds':0.,'backpressure_seconds':0.,'final_join_seconds':0.,'writer_service_seconds':0.,'maximum_pending':0,'tasks':0,'files':0,'bytes':0,'maximum_rss_kib':0}

    def check_memory(self):
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        with self.lock:self.metrics['maximum_rss_kib']=max(self.metrics['maximum_rss_kib'],peak)
        if peak>self.memory_limit_kib:raise MemoryError(f'PeakCPU RSS {peak}KiB exceeds frozen {self.memory_limit_kib}KiB')

    def _write(self,items):
        started=time.perf_counter();receipts=[]
        for path,arrays in items:
            self.check_memory()
            if self.fail_name is not None and path.name==self.fail_name:raise RuntimeError('Injected archive failure: '+self.fail_name)
            Z.save(path,**arrays);receipts.append({'path':str(path),'sha256':S.sha(path),'bytes':path.stat().st_size})
        with self.lock:self.metrics['writer_service_seconds']+=time.perf_counter()-started;self.metrics['files']+=len(receipts);self.metrics['bytes']+=sum(r['bytes'] for r in receipts)
        return receipts

    def _take_oldest(self):
        future=self.pending.popleft();self.results.extend(future.result())

    def submit(self,items):
        assert not self.closed;self.check_memory();self.metrics['tasks']+=1
        if not self.asynchronous:self.results.extend(self._write(items));return
        if len(self.pending)>=self.max_pending:
            before=time.perf_counter();self._take_oldest();self.metrics['backpressure_seconds']+=time.perf_counter()-before
        # Match np.save's layout choice exactly, including Fortran-contiguous input.
        before=time.perf_counter();owned=[]
        for path,arrays in items:
            copied={}
            for key,value in arrays.items():
                value=np.asanyarray(value);order='F' if value.flags.f_contiguous and not value.flags.c_contiguous else 'C';copied[key]=np.array(value,copy=True,order=order)
            owned.append((path,copied))
        self.metrics['ownership_copy_seconds']+=time.perf_counter()-before;self.check_memory();self.pending.append(self.pool.submit(self._write,owned));self.metrics['maximum_pending']=max(self.metrics['maximum_pending'],len(self.pending))
        # Surface completed errors at each producer boundary, before queuing more.
        while self.pending and self.pending[0].done():self._take_oldest()

    def close(self):
        if self.closed:return self.results
        before=time.perf_counter()
        try:
            while self.pending:self._take_oldest()
        finally:
            if self.pool:self.pool.shutdown(wait=True,cancel_futures=False)
            self.closed=True;self.metrics['final_join_seconds']+=time.perf_counter()-before;self.check_memory()
        return self.results
