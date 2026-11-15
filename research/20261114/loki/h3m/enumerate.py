"""Validate all N11 superqueen patterns before any multisize-source fit."""
import datetime as dt
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3m';OUT=ROOT/'artifacts/research/20261114/loki/h3m'
sys.path.insert(0,str(ROOT/'upstream'))
from cram import nqueens_data as D
spec=importlib.util.spec_from_file_location('super_enum',SRC.parent/'h3l/enumerate.py');E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    assert not(OUT/'feasibility.json').exists();OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    superqueens=E.enumerate_super(11);ordinary=np.asarray(D.completions(np.zeros(121,bool),11),bool)
    columns=ordinary.reshape(-1,11,11).argmax(2)
    keep=(np.abs(columns[:,1:]-columns[:,:-1])!=2).all(1)&(np.abs(columns[:,2:]-columns[:,:-2])!=1).all(1)
    assert {r.tobytes() for r in superqueens}=={r.tobytes() for r in ordinary[keep]}
    assert len({r.tobytes() for r in superqueens})==len(superqueens)
    for board in superqueens:
        cells=np.flatnonzero(board);assert len(cells)==11
        for a,b in itertools.combinations(cells,2):
            dr=abs(a//11-b//11);dc=abs(a%11-b%11)
            assert dr and dc and dr!=dc and (dr,dc) not in [(1,2),(2,1)]
    np.save(OUT/'solutions-n11.npy',superqueens)
    report=dict(status='validated_before_fitting',utc=dt.datetime.now(dt.timezone.utc).isoformat(),n=11,
        complete_superqueen_solutions=len(superqueens),complete_ordinary_queen_solutions=len(ordinary),
        independent_enumerations_match=True,every_board_pairwise_valid=True,no_cooccurrence_or_line_fit_inspected=True,
        seconds=time.perf_counter()-start,solutions_sha256=sha(OUT/'solutions-n11.npy'),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),SRC.parent/'h3l/enumerate.py',ROOT/'upstream/cram/nqueens_data.py']})
    (OUT/'feasibility.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
