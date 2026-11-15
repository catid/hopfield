"""Frozen compiled-BFS timing scope control on reused H2u graphs."""
import ctypes
import hashlib
import json
import platform
import subprocess
import time
import numpy as np
import study as S

DEST=S.OUT/'bfs_scope_control'
PP=S.HERE/'bfs-scope-control-protocol.json'


class Solver:
    def __init__(self,path):
        self.lib=ctypes.CDLL(str(path));self.fun=self.lib.solve_graphs;u8=ctypes.POINTER(ctypes.c_uint8);i32=ctypes.POINTER(ctypes.c_int32);self.fun.argtypes=[u8,ctypes.c_int,ctypes.c_int,ctypes.c_int,u8,i32,u8];self.fun.restype=ctypes.c_int
    def __call__(self,tokens):
        t=np.ascontiguousarray(tokens,dtype=np.uint8);assert t.ndim==3
        boards=np.empty_like(t);distances=np.empty(len(t),np.int32);status=np.empty(len(t),np.uint8);u8=ctypes.POINTER(ctypes.c_uint8);i32=ctypes.POINTER(ctypes.c_int32)
        rc=self.fun(t.ctypes.data_as(u8),*t.shape,boards.ctypes.data_as(u8),distances.ctypes.data_as(i32),status.ctypes.data_as(u8));assert rc==0,rc
        return boards,distances,status


def main():
    started=time.perf_counter();DEST.mkdir(parents=True,exist_ok=True);p=json.loads(PP.read_text());source=S.OUT/'window_fallback';assert S.sha(source/'datasets.json')==p['inputs']['manifest_sha256'];manifest=json.loads((source/'datasets.json').read_text());library=DEST/'bfs_solver.so';command=['cc','-O3','-std=c11','-shared','-fPIC',str(S.HERE/'bfs_scope_solver.c'),'-o',str(library)];subprocess.run(command,check=True);solver=Solver(library)
    fixtures=[]
    t=np.zeros((9,9),np.uint8);t[:5,0]=1;t[4,:5]=1;t[:5,4]=1;t[0,0]=2;t[0,4]=3;fixtures.append(('unique_detour',t,12))
    t=np.ones((9,9),np.uint8);t[0,0]=2;t[8,8]=3;fixtures.append(('tied_shortest_paths',t,16))
    t=np.zeros((9,9),np.uint8);t[0,0]=2;t[8,8]=3;fixtures.append(('disconnected',t,-1))
    t=np.ones((9,9),np.uint8);t[0,0]=2;fixtures.append(('missing_goal',t,-1));t=t.copy();t[1,1]=2;t[8,8]=3;fixtures.append(('duplicate_start',t,-1))
    fixture_boards,fixture_distances,fixture_status=solver(np.stack([x[1] for x in fixtures]));assert fixture_distances.tolist()==[x[2] for x in fixtures] and fixture_status.tolist()==[int(x[2]>=0) for x in fixtures]
    import audit as A
    for i,(_,tok,distance) in enumerate(fixtures):
        if distance>=0:assert A.independent(tok,fixture_boards[i].astype(bool))==(True,distance)
    warm_tokens=np.stack([S.maze(120,.2,np.random.SeedSequence([2026111597,i]))[0] for i in range(64)]);solver(warm_tokens)
    io=[];tokens=None
    for rep in range(3):
        begin=time.perf_counter();pieces=[]
        for ds in manifest['datasets']:
            with np.load(S.ROOT/ds['path']) as x:pieces.append(x['tokens'])
        tokens=np.concatenate(pieces);io.append(time.perf_counter()-begin)
    assert tokens.shape==(16384,120,120);rows=[];reference=None
    for rep in range(3):
        begin=time.perf_counter();outputs=solver(tokens);seconds=time.perf_counter()-begin
        if reference is None:reference=outputs
        else:assert all(np.array_equal(a,b) for a,b in zip(reference,outputs))
        rows.append({'repetition':rep,'solve_and_cpu_output_seconds':seconds,'seconds_per_graph':seconds/len(tokens),'exact_output_parity':True});S.emit('bfs_scope_timing',**rows[-1])
    np.savez_compressed(DEST/'outputs.npz',predictions=reference[0],distances=reference[1],status=reference[2])
    model='unknown'
    for line in open('/proc/cpuinfo'):
        if line.startswith('model name'):model=line.split(':',1)[1].strip();break
    result={'status':'measured_pending_independent_audit','utc':S.utc(),'host':platform.node(),'cpu_model':model,'single_threaded_solver':True,'gpu_jobs':0,'n_unique_reused_inputs':len(tokens),'repetitions':rows,'median_solve_and_cpu_output_seconds':float(np.median([r['solve_and_cpu_output_seconds'] for r in rows])),'archive_io_and_stacking_seconds':io,'io_cache_policy':'No OS cache flushing; separate from preloaded solver timing.','fixtures':[{'name':x[0],'expected_distance':x[2]} for x in fixtures],'warmup_seed':2026111597,'protocol_sha256':S.sha(PP),'script_sha256':S.sha(__file__),'c_source_sha256':S.sha(S.HERE/'bfs_scope_solver.c'),'library_sha256':S.sha(library),'compiler':subprocess.check_output(['cc','--version'],text=True).splitlines()[0],'compile_command':command,'outputs_sha256':S.sha(DEST/'outputs.npz'),'input_tokens_sha256':hashlib.sha256(tokens.tobytes()).hexdigest(),'total_worker_seconds':time.perf_counter()-started,'scope':'Ordinary hand-engineered BFS on suppliedgraph, no learning and no feedback into CRAM. Outputs and all3complete timings retained; archive I/O separately measured.'}
    S.save(DEST/'result.json',result);S.emit('bfs_scope_complete',median_seconds=result['median_solve_and_cpu_output_seconds'])


if __name__=='__main__':main()
