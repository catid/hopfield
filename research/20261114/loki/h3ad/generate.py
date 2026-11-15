"""Complete new target universes, independently checked within one CPU budget."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import itertools
import signal
import time
import numpy as np
from common import ROOT,SRC,OUT,sha,read,save,now,frozen,load

def queens(n):
    full=(1<<n)-1;answers=[]
    def recurse(columns,left,right,placed):
        if columns==full:answers.append(placed);return
        choices=full&~(columns|left|right)
        while choices:
            bit=choices&-choices;choices^=bit;recurse(columns|bit,(left|bit)<<1,(right|bit)>>1,placed+[bit.bit_length()-1])
    recurse(0,0,0,[]);return np.array(answers,np.uint8).reshape(-1,n)

def filter_leaper(columns,leaper):
    n=columns.shape[1];i,j=np.triu_indices(n,1);dr=j-i;dc=abs(columns[:,j].astype(np.int16)-columns[:,i].astype(np.int16));a,b=leaper;bad=((dr==a)&(dc==b))|((dr==b)&(dc==a));return columns[~bad.any(1)]

def main():
    p=frozen();assert not(OUT/'enumeration-gate.json').exists();OUT.mkdir(parents=True,exist_ok=True);start=time.process_time();wall=time.perf_counter();rows=[]
    def expired(*args):raise TimeoutError('Aggregate120CPU-second enumeration/verification ceiling')
    signal.signal(signal.SIGPROF,expired);signal.setitimer(signal.ITIMER_PROF,120)
    try:
        helper=load(SRC.parent/'h3y/common.py','h3ad_general_enumerator');ordinary={n:queens(n)for n in[12,13]};assert len(ordinary[12])==14200 and len(ordinary[13])==73712
        for bank,family in enumerate(p['families']):
            folder=OUT/f'b{bank}';folder.mkdir(parents=True,exist_ok=True);leaper=tuple(family['leaper']);source=np.load(ROOT/family['source_path']);independent_source=filter_leaper(ordinary[12],leaper);assert np.array_equal(source,independent_source)and len(source)==family['source_patterns']
            target=helper.bit_enumeration(13,[(0,1),(1,-1),(1,0),(1,1)],leapers=[leaper]);independent_target=filter_leaper(ordinary[13],leaper);assert np.array_equal(target,independent_target)and len(target)>0 and len(np.unique(target,axis=0))==len(target)
            y=np.zeros((len(target),169),bool);y[np.arange(len(y))[:,None],13*np.arange(13)[None,:]+target]=True;np.save(folder/'source-columns.npy',source);np.save(folder/'target-columns.npy',target);np.save(folder/'solutions-n13.npy',y)
            rows.append(dict(bank=bank,leaper=list(leaper),source_patterns=len(source),target_patterns=len(target),source_independent_complete=True,target_primary_independent_complete_identical=True,source_columns_sha256=sha(folder/'source-columns.npy'),target_columns_sha256=sha(folder/'target-columns.npy'),target_solutions_sha256=sha(folder/'solutions-n13.npy')));print('ENUMERATED',rows[-1],flush=True)
        signal.setitimer(signal.ITIMER_PROF,0);save(OUT/'enumeration-gate.json',dict(status='complete_verified',utc=now(),aggregate_cpu_seconds=time.process_time()-start,wall_seconds=time.perf_counter()-wall,budget_cpu_seconds=120,ordinary_queen_counts={str(n):len(v)for n,v in ordinary.items()},families=rows,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
    except BaseException as e:
        signal.setitimer(signal.ITIMER_PROF,0);save(OUT/'enumeration-gate.json',dict(status='incomplete_gate_failure',utc=now(),aggregate_cpu_seconds=time.process_time()-start,wall_seconds=time.perf_counter()-wall,budget_cpu_seconds=120,completed_family_rows=rows,error=repr(e),partial_universe_usable=False,protocol_sha256=sha(SRC/'protocol.json')));raise
    # Complete input enumeration starts only after both full solution universes pass.
    input_start=time.process_time();input_wall=time.perf_counter()
    for bank in range(2):
        folder=OUT/f'b{bank}';target=np.load(folder/'target-columns.npy');keys=[];cells=13*np.arange(13)[None,:]+target
        for clues in[1,2,3]:
            ix=np.array(list(itertools.combinations(range(13),clues)));multipliers=np.array([170**(2-j)for j in range(clues)],np.int64);part=((cells[:,ix]+1)*multipliers).sum(-1);keys.append(part.reshape(-1))
            literal=np.array([sum((int(cell)+1)*170**(2-j)for j,cell in enumerate(subset))for row in cells[:17]for subset in itertools.combinations(row,clues)],np.int64);assert np.array_equal(literal,part[:17].reshape(-1))
        unique,counts=np.unique(np.concatenate(keys),return_counts=True);assert len(unique)>=4096 and int(counts.sum())==len(target)*377;np.savez_compressed(folder/'input-universe.npz',keys=unique,counts=counts);save(folder/'universe.json',dict(status='complete_input_universe',unique_inputs=len(unique),all_input_solution_pairs=int(counts.sum()),complete_target_solutions=len(target),universe_sha256=sha(folder/'input-universe.npz'),enumeration_gate_sha256=sha(OUT/'enumeration-gate.json')));print('INPUT_UNIVERSE',bank,len(unique),len(target),flush=True)
    save(OUT/'input-preparation.json',dict(status='complete',aggregate_cpu_seconds=time.process_time()-input_start,wall_seconds=time.perf_counter()-input_wall,vectorized_subset_keys_checked_against_literal_first17_boards=True))
if __name__=='__main__':main()
