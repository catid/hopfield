"""Bounded queen-plus-camel feasibility with independent complete enumerations."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import datetime as dt
import hashlib
import itertools
import json
from pathlib import Path
import signal
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3t';OUT=ROOT/'artifacts/research/20261114/loki/h3t'
sys.path.insert(0,str(ROOT/'upstream'))
from cram import nqueens_data as D

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def enumerate_camel(n):
    full=(1<<n)-1;answers=[]
    def visit(columns,left,right,placed):
        if columns==full:
            board=np.zeros(n*n,bool);board[np.arange(n)*n+placed]=True;answers.append(board);return
        choices=full&~(columns|left|right)
        if placed:
            for c in [placed[-1]-3,placed[-1]+3]:
                if 0<=c<n:choices&=~(1<<c)
        if len(placed)>=3:
            for c in [placed[-3]-1,placed[-3]+1]:
                if 0<=c<n:choices&=~(1<<c)
        while choices:
            bit=choices&-choices;choices^=bit;visit(columns|bit,(left|bit)<<1,(right|bit)>>1,placed+[bit.bit_length()-1])
    visit(0,0,0,[]);return np.asarray(answers,bool).reshape(-1,n*n)
def main():
    assert not(OUT/'feasibility.json').exists();OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter();stage='source_enumeration'
    def timeout(*unused):raise TimeoutError('120-second CPU feasibility envelope reached')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
    try:
        enumerations={}
        for n in [12,13]:
            stage=f'N{n}_enumeration';began=time.perf_counter();solutions=enumerate_camel(n)
            ordinary=np.asarray(D.completions(np.zeros(n*n,bool),n),bool);columns=ordinary.reshape(-1,n,n).argmax(2)
            keep=(np.abs(columns[:,1:]-columns[:,:-1])!=3).all(1)&(np.abs(columns[:,3:]-columns[:,:-3])!=1).all(1)
            assert {r.tobytes() for r in solutions}=={r.tobytes() for r in ordinary[keep]}
            assert len(solutions)==len({r.tobytes() for r in solutions})
            for board in solutions:
                cells=np.flatnonzero(board);assert len(cells)==n
                for a,b in itertools.combinations(cells,2):
                    dr=abs(a//n-b//n);dc=abs(a%n-b%n)
                    assert dr and dc and dr!=dc and (dr,dc) not in [(1,3),(3,1)]
            np.save(OUT/f'solutions-n{n}.npy',solutions)
            enumerations[str(n)]=dict(complete_solutions=len(solutions),ordinary_queen_solutions=len(ordinary),independent_enumerations_match=True,
                every_board_pairwise_valid=True,seconds=time.perf_counter()-began,array_sha256=sha(OUT/f'solutions-n{n}.npy'))
            print(json.dumps(dict(stage=stage,**enumerations[str(n)])),flush=True)
        stage='input_universe';solutions=np.load(OUT/'solutions-n13.npy');positions=np.flatnonzero(solutions).reshape(len(solutions),13)%169
        counts=np.zeros(170**3,np.int32)
        for clues in [1,2,3]:
            subsets=np.array(list(itertools.combinations(range(13),clues)))
            for lo in range(0,len(positions),2048):
                cells=positions[lo:lo+2048,subsets]+1;keys=sum(cells[:,:,a].astype(np.int64)*170**(2-a) for a in range(clues)).ravel();np.add.at(counts,keys,1)
        keys=np.flatnonzero(counts);prior=np.zeros(0,np.int64);prior_hashes={}
        for study in ['h3j','h3k','h3l','h3m','h3n']:
            path=ROOT/f'artifacts/research/20261114/loki/{study}/data.npz';prior=np.union1d(prior,np.load(path)['test_keys']);prior_hashes[str(path.relative_to(ROOT))]=sha(path)
        eligible=keys[~np.isin(keys,prior)]
        np.savez_compressed(OUT/'input-universe.npz',keys=keys,counts=counts[keys],eligible_keys=eligible,prior_target_keys=prior)
        feasible=enumerations['12']['complete_solutions']>0 and len(eligible)>=4096
        report=dict(status='feasible_validated_no_fit' if feasible else 'infeasible_validated_no_fit',utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            task='queens plus camel attacks (1,3)/(3,1)',source_n=12,target_n=13,enumerations=enumerations,target_unique_input_universe=len(keys),
            prior_N13_search_input_keys=len(prior),excluded_overlap=len(keys)-len(eligible),eligible_fresh_inputs=len(eligible),requested_inputs=4096,
            no_support_counts_or_exposure_inspected=True,no_model_fits_or_search=True,seconds=time.perf_counter()-start,cpu_wall_seconds_envelope=120,
            input_universe_sha256=sha(OUT/'input-universe.npz'),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'upstream/cram/nqueens_data.py']},prior_data_sha256=prior_hashes)
        (OUT/'feasibility.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    except BaseException as exc:
        (OUT/'feasibility-failure.json').write_text(json.dumps(dict(status='failed',stage=stage,error=repr(exc),seconds=time.perf_counter()-start,cpu_wall_seconds_envelope=120),indent=2)+'\n');raise
    finally:signal.alarm(0)
if __name__=='__main__':main()
