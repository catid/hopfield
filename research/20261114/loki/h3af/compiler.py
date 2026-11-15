"""Model-only exact zero-F compilation with independent complete enumeration."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import signal
import time
import numpy as np
from common import ROOT,SRC,OUT,sha,read,save,now,frozen

def eligibility(m):
    q=int(m['q']);assert q==15;F=m['F'];J=m['J'];assert F.shape==J.shape==(q*q,q*q)and np.array_equal(F,F.T)and np.isin(F,[0,1]).all()and not F.diagonal().any();assert np.isfinite(J).all()and np.array_equal(J,J.T)and(J[F.astype(bool)]>0).all()and not J[~F.astype(bool)].any()and not m['h'].any()
    for row in range(q):
        cells=np.arange(q)+q*row;assert(F[np.ix_(cells,cells)]+np.eye(q)==1).all()
    for col in range(q):
        cells=q*np.arange(q)+col;assert(F[np.ix_(cells,cells)]+np.eye(q)==1).all()
    return dict(q=q,strict_positive_F_energy=True,complete_row_and_column_cliques=True,proof='At most one occupied cell per row/column; q=n implies exactly one per row/column. Exhaustive row-wise F-consistent recursion therefore enumerates every cardinality-q zero-energy state.')

def forward(F,q):
    blocked={(r,c):[sum(1<<j for j in range(q)if F[q*r+c,q*t+j])for t in range(q)]for r in range(q)for c in range(q)};answers=[];full=(1<<q)-1
    def visit(r,masks,columns):
        if r==q:answers.append(columns);return
        choices=full&~masks[r]
        while choices:
            bit=choices&-choices;choices^=bit;c=bit.bit_length()-1;new=masks[:]
            for t in range(r+1,q):new[t]|=blocked[r,c][t]
            visit(r+1,new,columns+[c])
    visit(0,[0]*q,[]);return np.array(answers,np.uint8).reshape(-1,q)

def reverse_literal(F,q):
    answers=[];selected=[];columns=[0]*q
    def visit(row):
        if row<0:answers.append(columns.copy());return
        for col in range(q):
            cell=q*row+col
            if any(F[cell,old]for old in selected):continue
            columns[row]=col;selected.append(cell);visit(row-1);selected.pop()
    visit(q-1);a=np.array(answers,np.uint8).reshape(-1,q)
    if len(a):a=a[np.lexsort(a.T[::-1])]
    return a

def main():
    p=frozen();from transfer import make;assert not(OUT/'compilation-gate.json').exists();OUT.mkdir(parents=True,exist_ok=True);start_cpu=time.process_time();start_wall=time.perf_counter();rows=[]
    def expired(*args):raise TimeoutError('Aggregate120CPU-second exact compilation/independent-proof ceiling')
    signal.signal(signal.SIGPROF,expired);signal.setitimer(signal.ITIMER_PROF,120)
    try:
        for bank,cfg in enumerate(p['models']):
            cpu=time.process_time();wall=time.perf_counter();folder=OUT/f'b{bank}';folder.mkdir(parents=True,exist_ok=True);path=folder/'model.npz';m,transfer=make(cfg,15);np.savez_compressed(path,**m);save(folder/'transfer.json',transfer);proof=eligibility(m);q=proof['q'];save(OUT/'compilation-progress.json',dict(bank=bank,stage='forward',utc=now()));primary_start=time.perf_counter();a=forward(m['F'],q);primary_seconds=time.perf_counter()-primary_start;np.save(folder/'unverified-forward-columns.npy',a);save(OUT/'compilation-progress.json',dict(bank=bank,stage='independent_reverse',utc=now(),unverified_forward_solutions=len(a),primary_seconds=primary_seconds));ind_start=time.perf_counter();b=reverse_literal(m['F'],q);independent_seconds=time.perf_counter()-ind_start;assert np.array_equal(a,b)and len(np.unique(a,axis=0))==len(a)
            boards=np.zeros((len(a),q*q),bool);cells=q*np.arange(q)[None,:]+a;boards[np.arange(len(a))[:,None],cells]=True;i,j=np.triu_indices(q*q,1);assert not((boards[:,i]&boards[:,j])@m['F'][i,j]).any()
            index=np.packbits(boards.T,axis=1,bitorder='little');np.savez_compressed(folder/'cache.npz',boards=boards,index=index,columns=a);(folder/'calibration-selection.json').write_bytes((ROOT/cfg['calibration_selection_path']).read_bytes())
            record=dict(bank=bank,status='complete_verified',utc=now(),zero_F_solutions=len(a),primary_seconds=primary_seconds,independent_seconds=independent_seconds,compile_wall_seconds=time.perf_counter()-wall,compile_cpu_seconds=time.process_time()-cpu,includes_source_transfer_model_checks_both_enumerators_index_and_serialization=True,proof=proof,source_model_sha256=sha(path),cache_sha256=sha(folder/'cache.npz'),protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__));save(folder/'compilation.json',record);rows.append(record);print('COMPILED',record,flush=True)
        signal.setitimer(signal.ITIMER_PROF,0);save(OUT/'compilation-gate.json',dict(status='complete_verified',utc=now(),aggregate_cpu_seconds=time.process_time()-start_cpu,wall_seconds=time.perf_counter()-start_wall,cpu_seconds_ceiling=120,families=rows,no_target_arrays_labels_keys_or_accuracy_accessed=True,protocol_sha256=sha(SRC/'protocol.json')))
    except BaseException as e:
        signal.setitimer(signal.ITIMER_PROF,0);save(OUT/'compilation-gate.json',dict(status='failed_or_incomplete',utc=now(),aggregate_cpu_seconds=time.process_time()-start_cpu,wall_seconds=time.perf_counter()-start_wall,completed_families=rows,error=repr(e),partial_cache_usable=False,progress=read(OUT/'compilation-progress.json')if(OUT/'compilation-progress.json').exists()else None,protocol_sha256=sha(SRC/'protocol.json')));raise
if __name__=='__main__':main()
