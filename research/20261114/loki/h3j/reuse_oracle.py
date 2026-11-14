"""Explicit exact-identity oracle reuse; no independent search is claimed."""
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3j';OUT=ROOT/'artifacts/research/20261114/loki/h3j'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def now():return dt.datetime.now(dt.timezone.utc).isoformat()


def main():
    audit=json.loads((OUT/'independent-preflight.json').read_text());assert audit['status']=='passed' and audit['oracle_solver_identity']['exact']
    assert not list((OUT/'search').glob('*/s*/result.json')),'Identity plan must precede every target outcome'
    locks=[]
    for seed in [0,1,2]:
        dest=OUT/f'search/true-graph-oracle/s{seed}';dest.mkdir(parents=True,exist_ok=True)
        lock=(dest/'job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
    plan=dict(status='ready',frozen_utc=now(),pid=os.getpid(),source_arm='line-extrapolation',derived_arm='true-graph-oracle',seeds=[0,1,2],
        actual_unique_search_jobs=6,avoided_identical_search_jobs=3,protocol_sha256=sha(SRC/'protocol.json'),
        identity_receipt_sha256=sha(OUT/'independent-preflight.json'),source_sha256=sha(__file__),
        rationale='All solver-used parameters and every search input/seed are equal before inference. Oracle outputs are exact mappings from extrapolation, not independent accuracy replications.',
        original_protocol='The frozen protocol lists9 jobs. This pre-inference execution amendment deduplicates3 exactly identical oracle jobs without changing any arm or hypothesis.')
    save(OUT/'oracle-identity-plan.json',plan);print(json.dumps(plan),flush=True)
    mappings=[];started=time.monotonic()
    for seed,lock in zip([0,1,2],locks):
        source=OUT/f'search/line-extrapolation/s{seed}';target=OUT/f'search/true-graph-oracle/s{seed}'
        while not(source/'result.json').exists():
            if time.monotonic()-started>600:raise RuntimeError('Source search unavailable within600seconds; no oracle output derived')
            time.sleep(.2)
        rec=json.loads((source/'result.json').read_text());assert rec['status']=='complete' and rec['records_sha256']==sha(source/'records.npz')
        os.link(source/'records.npz',target/'records.npz')
        model=OUT/'models/true-graph-oracle/model.pt'
        derived=dict(rec,status='derived_by_exact_identity',utc=now(),arm='true-graph-oracle',search_seconds=0.,
            model_path=str(model.relative_to(ROOT)),model_sha256=sha(model),shared_source_search_seconds=rec['search_seconds'],
            source_result_path=str((source/'result.json').relative_to(ROOT)),source_result_sha256=sha(source/'result.json'),
            identity_plan_sha256=sha(OUT/'oracle-identity-plan.json'),derivation_source_sha256=sha(__file__),independent_search=False)
        save(target/'result.json',derived)
        mappings.append(dict(seed=seed,source_result=str((source/'result.json').relative_to(ROOT)),source_result_sha256=sha(source/'result.json'),
            derived_result=str((target/'result.json').relative_to(ROOT)),derived_result_sha256=sha(target/'result.json'),records_sha256=sha(target/'records.npz')))
        lock.close();print(json.dumps(dict(event='oracle_derived',seed=seed)),flush=True)
    save(OUT/'oracle-identity-mapping.json',dict(status='complete',completed_utc=now(),plan_sha256=sha(OUT/'oracle-identity-plan.json'),mappings=mappings))


if __name__=='__main__':main()
