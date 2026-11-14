"""Reserve an unconditioned unique-input N13 evaluation cohort, without model use."""
import datetime as dt
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3j';OUT=ROOT/'artifacts/research/20261114/loki/h3j'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    start=time.perf_counter();assert not(OUT/'dataset.json').exists()
    solutions=np.load(OUT/'solutions.npy');assert solutions.shape==(73712,169)
    positions=np.array([np.flatnonzero(row) for row in solutions]);assert positions.shape==(73712,13)
    assert len({row.tobytes() for row in solutions})==73712
    for row in positions:
        r=row//13;c=row%13
        assert len(set(r))==len(set(c))==len(set(r-c))==len(set(r+c))==13
    # Dense integer key space avoids materializing 27.8 million repeated rows.
    counts=np.zeros(170**3,dtype=np.int32)
    for k in [1,2,3]:
        subsets=np.array(list(itertools.combinations(range(13),k)))
        for lo in range(0,len(positions),512):
            cells=positions[lo:lo+512,subsets]+1
            keys=sum(cells[:,:,j].astype(np.int64)*170**(2-j) for j in range(k)).ravel()
            np.add.at(counts,keys,1)
    pool=np.flatnonzero(counts);selected=np.sort(np.random.default_rng(2026111417).permutation(pool)[:4096])
    inputs=np.zeros((len(selected),169),bool)
    for power in [2,1,0]:
        cells=selected//170**power%170;present=cells>0;inputs[np.flatnonzero(present),cells[present]-1]=True
    for given,number in zip(inputs,counts[selected]):assert solutions[:,given].all(1).sum()==number>0
    np.savez_compressed(OUT/'data.npz',all_solutions=solutions,test_keys=selected,test_inputs=inputs,test_solution_counts=counts[selected],
        full_universe_keys=pool,full_universe_counts=counts[pool])
    pool_clues=(pool//170**2>0).astype(int)+(pool//170%170>0)+(pool%170>0)
    rec=dict(status='reserved_before_target_models_or_search',utc=dt.datetime.now(dt.timezone.utc).isoformat(),n=13,
        complete_solutions=73712,input_seed=2026111417,unique_input_universe=len(pool),test_unique=len(selected),
        universe_by_clues={str(k):int((pool_clues==k).sum()) for k in [1,2,3]},
        test_by_clues={str(k):int((inputs.sum(1)==k).sum()) for k in [1,2,3]},all_compatible_test_completions=int(counts[selected].sum()),
        distribution='Uniform sample of unique valid one/two/three-clue inputs from all73712 N13 solutions. No compatibility conditioning is needed because all training patterns are N12.',
        target_label_use='Target solutions are used only for input reservation, exhaustive answer counts and evaluation; none enters model construction, line selection, thresholds or temperature.',
        construction_seconds=time.perf_counter()-start,data_sha256=sha(OUT/'data.npz'),solution_array_sha256=sha(OUT/'solutions.npy'),
        source_sha256=sha(__file__),source_enumerator_sha256=sha(ROOT/'upstream/cram/nqueens_data.py'))
    (OUT/'dataset.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec))


if __name__=='__main__':main()
