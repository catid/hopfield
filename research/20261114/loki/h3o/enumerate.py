"""Bounded, independently checked N15 superqueen enumeration; no model fitting."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import datetime as dt
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import signal
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3o';OUT=ROOT/'artifacts/research/20261114/loki/h3o'
sys.path.insert(0,str(ROOT/'upstream'))
from cram import nqueens_data as D
spec=importlib.util.spec_from_file_location('super_enum',SRC.parent/'h3l/enumerate.py');E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def emit(**record):print(json.dumps(record),flush=True)
def main():
    assert not(OUT/'feasibility.json').exists();OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter();stage='direct_enumeration'
    def timeout(*unused):raise TimeoutError('300-second CPU feasibility envelope reached')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(300)
    try:
        superqueens=E.enumerate_super(15);emit(stage=stage,complete_solutions=len(superqueens),seconds=time.perf_counter()-start)
        stage='independent_ordinary_enumeration';ordinary=np.asarray(D.completions(np.zeros(225,bool),15),bool)
        emit(stage=stage,complete_solutions=len(ordinary),seconds=time.perf_counter()-start)
        columns=ordinary.reshape(-1,15,15).argmax(2)
        keep=(np.abs(columns[:,1:]-columns[:,:-1])!=2).all(1)&(np.abs(columns[:,2:]-columns[:,:-2])!=1).all(1)
        assert {r.tobytes() for r in superqueens}=={r.tobytes() for r in ordinary[keep]}
        assert len({r.tobytes() for r in superqueens})==len(superqueens);stage='pairwise_validation'
        for board in superqueens:
            cells=np.flatnonzero(board);assert len(cells)==15
            for a,b in itertools.combinations(cells,2):
                dr=abs(a//15-b//15);dc=abs(a%15-b%15)
                assert dr and dc and dr!=dc and (dr,dc) not in [(1,2),(2,1)]
        np.save(OUT/'solutions-n15.npy',superqueens)
        report=dict(status='validated_before_fitting',utc=dt.datetime.now(dt.timezone.utc).isoformat(),n=15,
            complete_superqueen_solutions=len(superqueens),complete_ordinary_queen_solutions=len(ordinary),independent_enumerations_match=True,
            every_board_pairwise_valid=True,no_cooccurrence_or_line_fit_inspected=True,seconds=time.perf_counter()-start,cpu_wall_seconds_envelope=300,
            solutions_sha256=sha(OUT/'solutions-n15.npy'),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),SRC.parent/'h3l/enumerate.py',ROOT/'upstream/cram/nqueens_data.py']})
        (OUT/'feasibility.json').write_text(json.dumps(report,indent=2)+'\n');emit(**report)
    except BaseException as exc:
        rec=dict(status='failed',stage=stage,error=repr(exc),seconds=time.perf_counter()-start,cpu_wall_seconds_envelope=300,source_sha256=sha(__file__))
        (OUT/'feasibility-failure.json').write_text(json.dumps(rec,indent=2)+'\n');raise
    finally:signal.alarm(0)
if __name__=='__main__':main()
