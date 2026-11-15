"""Paired full compilation, changing only exact certificate implementation."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import resource
import signal
import time
import numpy as np
from common import ROOT,SRC,OUT,sha,read,save,now,frozen

def cpu():
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN);return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime

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



def compile_one(bank,arm,cfg,binary_sha):
    from transfer import make
    from cpp_verifier import run
    folder=OUT/f'b{bank}/{arm}';folder.mkdir(parents=True,exist_ok=True);wall=time.perf_counter();start_cpu=cpu();m,transfer=make(cfg,15);np.savez_compressed(folder/'model.npz',**m);save(folder/'transfer.json',transfer);proof=eligibility(m);q=15;save(OUT/'progress.json',dict(bank=bank,arm=arm,stage='forward',utc=now()));start=time.perf_counter();a=forward(m['F'],q);forward_seconds=time.perf_counter()-start;np.save(folder/'unverified-forward-columns.npy',a);save(OUT/'progress.json',dict(bank=bank,arm=arm,stage='compiled_reverse',utc=now()));start=time.perf_counter();b=run(m['F'],q,f'N15-b{bank}-{arm}');reverse_seconds=time.perf_counter()-start;assert np.array_equal(a,b)and len(np.unique(a,axis=0))==len(a)
    save(OUT/'progress.json',dict(bank=bank,arm=arm,stage='certificate',utc=now()));start=time.perf_counter();boards=np.zeros((len(a),q*q),bool);cells=q*np.arange(q)[None,:]+a;boards[np.arange(len(a))[:,None],cells]=True
    if arm=='dense':
        i,j=np.triu_indices(q*q,1);counts=(boards[:,i]&boards[:,j])@m['F'][i,j]
    else:
        i,j=np.triu_indices(q,1);counts=m['F'][cells[:,i],cells[:,j]].sum(1,dtype=np.int64)
    assert not counts.any();certificate_seconds=time.perf_counter()-start;index=np.packbits(boards.T,axis=1,bitorder='little');np.savez_compressed(folder/'cache.npz',boards=boards,index=index,columns=a)
    r=dict(bank=bank,arm=arm,status='complete_verified',utc=now(),zero_F_solutions=len(a),forward_seconds=forward_seconds,reverse_seconds=reverse_seconds,certificate_seconds=certificate_seconds,compile_wall_seconds=time.perf_counter()-wall,compile_cpu_seconds=cpu()-start_cpu,model_sha256=sha(folder/'model.npz'),cache_sha256=sha(folder/'cache.npz'),compiled_verifier_binary_sha256=binary_sha,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__),proof=proof);save(folder/'compilation.json',r);print('COMPILED',bank,arm,r['compile_wall_seconds'],flush=True);return r

def main():
    from cpp_verifier import prerequisites
    p=frozen();assert not(OUT/'compiler-result.json').exists();OUT.mkdir(parents=True,exist_ok=True);wall=time.perf_counter();start_cpu=cpu();rows=[];prerequisite=None
    def expired(*args):raise TimeoutError('H3ai120CPU-second/180wall-second gate')
    signal.signal(signal.SIGUSR1,expired)
    try:
        prerequisite=prerequisites(reverse_literal,forward,cpu)
        for bank,cfg in enumerate(p['models']):
            order=['dense','occupied']if bank==0 else['occupied','dense'];pair=[]
            for arm in order:pair.append(compile_one(bank,arm,cfg,prerequisite['binary_sha256']))
            assert pair[0]['model_sha256']==pair[1]['model_sha256']and pair[0]['cache_sha256']==pair[1]['cache_sha256']and pair[0]['zero_F_solutions']==pair[1]['zero_F_solutions'];rows.append(dict(bank=bank,family=cfg['family'],order=order,arms=pair,full_cache_and_model_byte_identity=True))
        save(OUT/'compiler-result.json',dict(status='complete_verified',utc=now(),rows=rows,prerequisite=prerequisite,compiler_wall_seconds=time.perf_counter()-wall,compiler_cpu_seconds=cpu()-start_cpu,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
    except BaseException as e:
        save(OUT/'compiler-result.json',dict(status='failed_or_incomplete',utc=now(),completed_pairs=rows,prerequisite=prerequisite,error=repr(e),compiler_wall_seconds=time.perf_counter()-wall,compiler_cpu_seconds=cpu()-start_cpu,protocol_sha256=sha(SRC/'protocol.json')));raise
if __name__=='__main__':main()
