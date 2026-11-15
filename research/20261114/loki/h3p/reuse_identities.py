"""Map all prespecified logical models through exact pre-inference energy identity."""
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3p';OUT=ROOT/'artifacts/research/20261114/loki/h3p'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    audit=json.loads((OUT/'independent-preflight.json').read_text());assert audit['status']=='passed' and audit['all_identities_exact']
    assert not list((OUT/'search').glob('*/s*/result.json'))
    groups=json.loads((OUT/'energy-groups.json').read_text());pending=[]
    for group in groups['groups']:
        for arm in group['members']:
            if arm==group['representative']:continue
            for seed in [0,1,2]:
                dest=OUT/f'search/{arm}/s{seed}';dest.mkdir(parents=True,exist_ok=True)
                lock=(dest/'job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                pending.append(dict(source_arm=group['representative'],target_arm=arm,seed=seed,lock=lock))
    plan=dict(status='ready',frozen_utc=now(),pid=os.getpid(),groups=groups['groups'],actual_unique_search_jobs=groups['actual_unique_search_jobs'],
        avoided_identical_search_jobs=groups['avoided_identical_jobs'],protocol_sha256=sha(SRC/'protocol.json'),
        identity_receipt_sha256=sha(OUT/'independent-preflight.json'),energy_groups_sha256=sha(OUT/'energy-groups.json'),source_sha256=sha(__file__),
        rationale='Every prespecified source arm remains represented. Equal S,B,q,delta,tau,unit plus equal inputs and seeds imply identical outputs. Mappings are not independent search replication.')
    save(OUT/'identity-plan.json',plan);print(json.dumps(plan),flush=True);mappings=[];started=time.monotonic()
    while pending:
        if time.monotonic()-started>600:raise RuntimeError('Identity source unavailable within600seconds; no substitute output generated')
        remaining=[]
        for item in pending:
            source=OUT/f"search/{item['source_arm']}/s{item['seed']}";target=OUT/f"search/{item['target_arm']}/s{item['seed']}"
            if not(source/'result.json').exists():remaining.append(item);continue
            rec=json.loads((source/'result.json').read_text());assert rec['status']=='complete' and rec['records_sha256']==sha(source/'records.npz')
            os.link(source/'records.npz',target/'records.npz');model=OUT/f"models/{item['target_arm']}/model.pt"
            derived=dict(rec,status='derived_by_exact_identity',utc=now(),arm=item['target_arm'],search_seconds=0.,
                model_path=str(model.relative_to(ROOT)),model_sha256=sha(model),shared_source_search_seconds=rec['search_seconds'],
                source_result_path=str((source/'result.json').relative_to(ROOT)),source_result_sha256=sha(source/'result.json'),
                identity_plan_sha256=sha(OUT/'identity-plan.json'),derivation_source_sha256=sha(__file__),independent_search=False)
            save(target/'result.json',derived)
            mappings.append(dict(seed=item['seed'],source_arm=item['source_arm'],target_arm=item['target_arm'],
                source_result=str((source/'result.json').relative_to(ROOT)),source_result_sha256=sha(source/'result.json'),
                derived_result=str((target/'result.json').relative_to(ROOT)),derived_result_sha256=sha(target/'result.json'),records_sha256=sha(target/'records.npz')))
            item['lock'].close()
        pending=remaining
        if pending:time.sleep(.2)
    save(OUT/'identity-mapping.json',dict(status='complete',completed_utc=now(),plan_sha256=sha(OUT/'identity-plan.json'),mappings=mappings))
    print(json.dumps(dict(status='complete',mapped_jobs=len(mappings))),flush=True)


if __name__=='__main__':main()
