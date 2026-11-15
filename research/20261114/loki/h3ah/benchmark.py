"""Count equality and paired runtime for dense versus occupied-pair certificates."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib
import itertools
import resource
import signal
import time
import numpy as np
from common import ROOT,SRC,OUT,sha,read,save,now,frozen

def cpu():
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN);return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime

def dense(boards,F):
    i,j=np.triu_indices(225,1)
    return (boards[:,i]&boards[:,j])@F[i,j]

def occupied(cells,F):
    i,j=np.triu_indices(15,1)
    return F[cells[:,i],cells[:,j]].sum(1,dtype=np.int64)

def from_ready(ready,kind):return 15*np.arange(15)[None,:]+ready if kind=='row_columns' else ready

def solve(arm,path,ready,kind,boards,F):
    if path=='available_columns_or_indices':
        cells=from_ready(ready,kind)
        if arm=='occupied':return occupied(cells,F)
        converted=np.zeros((len(cells),225),bool);converted[np.arange(len(cells))[:,None],cells]=True;return dense(converted,F)
    if arm=='dense':return dense(boards,F)
    # Full input extraction is inside the measured optimized call.
    cells=np.nonzero(boards)[1].reshape(-1,15)
    return occupied(cells,F)

def digest(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()

def main():
    p=frozen();assert not(OUT/'benchmark-result.json').exists();OUT.mkdir(parents=True,exist_ok=True);started=time.perf_counter();start_cpu=cpu();rows=[];datasets=[]
    def expired(*args):raise TimeoutError('H3ah120aggregateCPU-second/180wall-second gate')
    signal.signal(signal.SIGUSR1,expired)
    try:
        for bank,cfg in enumerate(p['models']):
            m=np.load(ROOT/cfg['model_path']);cache=np.load(ROOT/cfg['cache_path']);F=m['F'];boards=cache['boards'];columns=cache['columns'];assert F.dtype==np.int32 and np.isin(F,[0,1]).all();assert(boards.sum(1)==15).all()and np.array_equal(np.sort(columns,axis=1),np.repeat(np.arange(15)[None,:],len(columns),axis=0))
            rng=np.random.default_rng(p['random_grid']['seeds'][bank]);random_cells=np.stack([np.sort(rng.choice(225,15,replace=False))for _ in range(8192)]);special=np.array([np.arange(15),15*np.arange(15),16*np.arange(15)],np.int64);allcells=np.concatenate([random_cells,special]);random_boards=np.zeros((len(allcells),225),bool);random_boards[np.arange(len(allcells))[:,None],allcells]=True;assert(random_boards.sum(1)==15).all()
            for name,ready,kind,data in [('complete_cache',columns,'row_columns',boards),('arbitrary_cardinality15',allcells,'occupied_cell_indices',random_boards)]:
                folder=OUT/f'b{bank}/{name}';folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'inputs.npz',boards=data,ready=ready);d=dict(bank=bank,name=name,states=len(data),ready_kind=kind,boards_sha256=digest(data),ready_sha256=digest(ready),input_npz_sha256=sha(folder/'inputs.npz'),model_sha256=sha(ROOT/cfg['model_path']));datasets.append(d)
                # Independent scalar reference, specified sample, before timed trials.
                literal=np.array([sum(int(F[a,b])for a,b in itertools.combinations(np.flatnonzero(board),2))for board in data[:128]],np.int64)
                fast=occupied(from_ready(ready[:128],kind),F);assert np.array_equal(literal,fast)
                counts=None
                for path in ['available_columns_or_indices','full_board']:
                    times={'dense':[],'occupied':[]};pairs=[]
                    for trial in range(3):
                        # Alternate first method across trials, paths, datasets and families.
                        parity=(trial+bank+int(name=='arbitrary_cardinality15')+int(path=='full_board'))%2;order=['dense','occupied']if parity==0 else['occupied','dense'];results={};pair=dict(trial=trial,order=order,times={})
                        for arm in order:
                            save(OUT/'progress.json',dict(utc=now(),bank=bank,dataset=name,path=path,trial=trial,arm=arm,completed_pairs=len(rows)))
                            wall=time.perf_counter();c=cpu();result=solve(arm,path,ready,kind,data,F);elapsed=time.perf_counter()-wall;elapsed_cpu=cpu()-c;assert result.dtype.kind in 'iu';results[arm]=result;times[arm].append(elapsed);pair['times'][arm]=dict(wall_seconds=elapsed,cpu_seconds=elapsed_cpu,integer_count_sha256=digest(result.astype(np.int64)))
                        assert np.array_equal(results['dense'],results['occupied'])and np.array_equal(results['occupied'][:128],literal)
                        if counts is None:counts=results['occupied'];np.save(folder/'conflict-counts.npy',counts)
                        else:assert np.array_equal(counts,results['occupied'])
                        pair['all_integer_counts_equal']=True;pairs.append(pair)
                    ratio=float(np.median(times['dense'])/np.median(times['occupied']));row=dict(bank=bank,family=cfg['family'],dataset=name,input_path=path,states=len(data),ready_kind=kind,all_integer_counts_equal=True,dense_median_seconds=float(np.median(times['dense'])),occupied_median_seconds=float(np.median(times['occupied'])),speed_ratio=ratio,required_speed_ratio=20.,decision='supported'if ratio>=20 else'falsified',zero_conflict_states=int((counts==0).sum()),nonzero_conflict_states=int((counts>0).sum()),maximum_conflicts=int(counts.max()),count_sha256=digest(counts),trials=pairs);rows.append(row);print('PAIRED',bank,name,path,ratio,flush=True)
                if name=='complete_cache':assert not counts.any()
                else:assert(counts>0).any()and np.array_equal(counts[-3:],np.array([105,105,105]))
        save(OUT/'benchmark-result.json',dict(status='complete_verified',utc=now(),decision='supported'if all(r['decision']=='supported'for r in rows)else'falsified',rows=rows,datasets=datasets,benchmark_cpu_seconds=cpu()-start_cpu,benchmark_wall_seconds=time.perf_counter()-started,all_integer_counts_equal=True,representation_conversion_included=True,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
    except BaseException as e:
        save(OUT/'benchmark-result.json',dict(status='failed_or_incomplete',utc=now(),error=repr(e),completed_rows=rows,datasets=datasets,benchmark_cpu_seconds=cpu()-start_cpu,benchmark_wall_seconds=time.perf_counter()-started,protocol_sha256=sha(SRC/'protocol.json')));raise
if __name__=='__main__':main()
