"""Superqueen feasibility only: two independent complete enumerations; no fit."""
import datetime as dt
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3l';OUT=ROOT/'artifacts/research/20261114/loki/h3l'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def enumerate_super(n):
    full=(1<<n)-1;answers=[]
    def search(columns,left,right,placed):
        if columns==full:
            board=np.zeros(n*n,bool);board[np.arange(n)*n+placed]=True;answers.append(board);return
        choices=full&~(columns|left|right)
        if placed:
            for col in [placed[-1]-2,placed[-1]+2]:
                if 0<=col<n:choices&=~(1<<col)
        if len(placed)>1:
            for col in [placed[-2]-1,placed[-2]+1]:
                if 0<=col<n:choices&=~(1<<col)
        while choices:
            bit=choices&-choices;choices^=bit
            search(columns|bit,(left|bit)<<1,(right|bit)>>1,placed+[bit.bit_length()-1])
    search(0,0,0,[])
    return np.asarray(answers,dtype=bool).reshape(-1,n*n)


def main():
    assert not(OUT/'feasibility.json').exists();OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    ip=ROOT/'artifacts/research/20261114/loki/h3i/data.npz';jp=ROOT/'artifacts/research/20261114/loki/h3j/data.npz';kp=ROOT/'artifacts/research/20261114/loki/h3k/data.npz'
    i=dict(np.load(ip));j=dict(np.load(jp));k=dict(np.load(kp));reports={}
    for n,ordinary in [(12,i['all_solutions']),(13,j['all_solutions'])]:
        began=time.perf_counter();sol=enumerate_super(n)
        # Independent exhaustive method: filter the previously validated full queen universe.
        columns=ordinary.reshape(-1,n,n).argmax(2)
        keep=(np.abs(columns[:,1:]-columns[:,:-1])!=2).all(1)&(np.abs(columns[:,2:]-columns[:,:-2])!=1).all(1)
        filtered=ordinary[keep]
        assert {row.tobytes() for row in sol}=={row.tobytes() for row in filtered}
        assert len({row.tobytes() for row in sol})==len(sol)
        for row in sol:
            cells=np.flatnonzero(row);assert len(cells)==n
            for a,b in itertools.combinations(cells,2):
                dr=abs(a//n-b//n);dc=abs(a%n-b%n)
                assert dr and dc and dr!=dc and (dr,dc) not in [(1,2),(2,1)]
        np.save(OUT/f'solutions-n{n}.npy',sol)
        reports[str(n)]=dict(complete_solutions=len(sol),ordinary_queen_universe=len(ordinary),
            independent_enumerations_identical=True,every_board_pairwise_valid=True,seconds=time.perf_counter()-began,
            array_sha256=sha(OUT/f'solutions-n{n}.npy'))
    sol=np.load(OUT/'solutions-n13.npy');positions=np.array([np.flatnonzero(row) for row in sol]);counts=np.zeros(170**3,dtype=np.int32)
    for clues in [1,2,3]:
        subsets=np.array(list(itertools.combinations(range(13),clues)))
        cells=positions[:,subsets]+1;keys=sum(cells[:,:,a].astype(np.int64)*170**(2-a) for a in range(clues)).ravel()
        np.add.at(counts,keys,1)
    universe=np.flatnonzero(counts);prior=np.union1d(j['test_keys'],k['test_keys']);eligible=universe[~np.isin(universe,prior)]
    clue_count=lambda keys:(keys//170**2>0).astype(int)+(keys//170%170>0)+(keys%170>0)
    np.savez_compressed(OUT/'input-universe.npz',keys=universe,counts=counts[universe],eligible_keys=eligible,prior_target_keys=prior)
    report=dict(status='feasible_enumerated_no_model_fitted',utc=dt.datetime.now(dt.timezone.utc).isoformat(),task='queen+knight attacks',
        enumerations=reports,source_bank_proposed=min(128,reports['12']['complete_solutions']),target_requested_inputs=4096,
        target_unique_input_universe=len(universe),prior_input_overlap=len(universe)-len(eligible),target_eligible_new_inputs=len(eligible),
        eligible_by_clues={str(c):int((clue_count(eligible)==c).sum()) for c in [1,2,3]},
        no_support_counts_or_line_fits=True,no_inference_outcomes=True,total_seconds=time.perf_counter()-start,
        input_universe_sha256=sha(OUT/'input-universe.npz'),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ip,jp,kp]})
    (OUT/'feasibility.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
