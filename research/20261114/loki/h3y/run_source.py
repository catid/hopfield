"""Source-only H3y learner and bounded two-process CPU campaign."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import concurrent.futures
import datetime as dt
import fcntl
import hashlib
import itertools
import json
import signal
import socket
import subprocess
import sys
import time
import numpy as np
from common import ROOT,SRC,OUT,sha,save,read,line_enumeration,bit_enumeration,counts_and_lines,candidate_orbits,bits
from verify_source import verify

def source_case(case_id):
    assert socket.gethostname()=='loki';folder=OUT/f'cases/{case_id}';folder.mkdir(parents=True,exist_ok=True);assert not(folder/'result.json').exists()
    p=read(SRC/'protocol.json');generation=read(OUT/'generation.json');assert generation['status']=='complete'and generation['protocol_sha256']==sha(SRC/'protocol.json')
    for rel,h in p['source_sha256'].items():assert sha(ROOT/rel)==h
    source=next(r for r in generation['cases']if r['id']==case_id);assert sha(ROOT/source['source_path'])==source['source_sha256'];n=source['n']
    started=time.perf_counter();cpu=time.process_time();stage='load_source'
    result=dict(id=case_id,n=n,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=source['source_sha256'],supplied_complete_source=True,no_target_file_access=True)
    def expired(*unused):raise TimeoutError('Frozen60second per-case source inference and certificate budget')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,60)
    try:
        columns=np.load(ROOT/source['source_path']);result['source_patterns']=len(columns)
        assert columns.shape==(source['source_patterns'],n)
        if len(columns)==0:
            result.update(status='empty_source',source_structural_failure=True,minimum_cardinality=None,minima=[])
        else:
            stage='source_lines';counts,lines=counts_and_lines(columns,n);result['inferred_lines']=lines
            if (0,1)not in lines or(1,0)not in lines:
                result.update(status='ineligible_row_column_inference',source_structural_failure=True,minimum_cardinality=None,minima=[])
            else:
                stage='line_compatible_universe';key=hashlib.sha256(json.dumps(dict(n=n,lines=lines),sort_keys=True).encode()).hexdigest();cache=OUT/f'cache/{key}';cache.mkdir(parents=True,exist_ok=True)
                with(cache/'cache.lock').open('a')as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    reused=(cache/'receipt.json').exists()
                    if not reused:
                        a=line_enumeration(n,lines);b=bit_enumeration(n,lines);assert np.array_equal(a,b);np.save(cache/'primary.npy',a);np.save(cache/'independent.npy',b)
                        save(cache/'receipt.json',dict(n=n,lines=lines,primary_sha256=sha(cache/'primary.npy'),independent_sha256=sha(cache/'independent.npy'),constructor_sha256=sha(SRC/'common.py'),boards=len(a)))
                    receipt=read(cache/'receipt.json');assert receipt['n']==n and receipt['lines']==[list(x)for x in lines]and receipt['constructor_sha256']==sha(SRC/'common.py')
                    assert sha(cache/'primary.npy')==receipt['primary_sha256']and sha(cache/'independent.npy')==receipt['independent_sha256']
                    universe=np.load(cache/'primary.npy');independent=np.load(cache/'independent.npy')
                keys={tuple(map(int,row))for row in columns};positive=np.array([tuple(map(int,row))in keys for row in universe]);assert int(positive.sum())==len(columns)==len(keys)
                generators,orbits=candidate_orbits(counts,lines);incidence=np.zeros((len(universe),len(orbits)),bool)
                for k,orbit in enumerate(orbits):
                    for dr,dc in orbit:incidence[:,k]|=(universe[:,dr:].astype(int)-universe[:,:n-dr].astype(int)==dc).any(1)
                assert not incidence[positive].any();neg=np.flatnonzero(~positive);coverbits=[bits(incidence[neg,k])for k in range(len(orbits))];allbits=(1<<len(neg))-1
                stage='minimum_cardinality_search';checked=[];minima=[];minimum=None
                for size in range(len(orbits)+1):
                    for chosen in itertools.combinations(range(len(orbits)),size):
                        covered=0
                        for k in chosen:covered|=coverbits[k]
                        missing=allbits^covered;witness=int(neg[(missing&-missing).bit_length()-1])if missing else None
                        checked.append(dict(indices=list(chosen),cardinality=size,rejected_negatives=covered.bit_count(),covers_all_negatives=missing==0,
                                            first_uncovered_universe_index=witness,first_uncovered_columns=universe[witness].astype(int).tolist()if witness is not None else None))
                        if not missing:minima.append(list(chosen))
                    if minima:minimum=size;break
                if minimum is None:
                    result.update(status='no_source_cover',source_structural_failure=True,minimum_cardinality=None,minima=[])
                else:
                    stage='independent_minimum_certificate';certificate=verify(columns,n,universe,independent,lines,generators,incidence,positive,checked,minima,minimum)
                    np.savez_compressed(folder/'certificate.npz',source_columns=columns,universe_columns=universe,is_positive=positive,incidence=incidence)
                    save(folder/'verification.json',certificate);save(folder/'checked-subsets.json',checked)
                    result.update(status='source_certified',source_structural_failure=len(minima)!=1,source_universe_boards=len(universe),source_negatives=len(neg),
                                  candidate_generators=generators,candidate_orbits=orbits,minimum_cardinality=minimum,minima=minima,
                                  selected_generator_sets=[[generators[k]for k in chosen]for chosen in minima],unique_minimum=len(minima)==1,
                                  checked_subset_count=len(checked),certificate_sha256=sha(folder/'certificate.npz'),verification_sha256=sha(folder/'verification.json'),
                                  checked_subsets_sha256=sha(folder/'checked-subsets.json'),cache_key=key,cache_reused=reused,cache_receipt=receipt)
        elapsed=time.perf_counter()-started;cpu_elapsed=time.process_time()-cpu;assert elapsed<60 and cpu_elapsed<60
        result.update(wall_seconds=elapsed,cpu_seconds=cpu_elapsed,completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    except BaseException as exc:
        result.update(status='incomplete_timeout'if isinstance(exc,TimeoutError)else'incomplete_validation_error',error=repr(exc),stage=stage,
                      wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    finally:signal.setitimer(signal.ITIMER_REAL,0)
    save(folder/'result.json',result);print(json.dumps({k:result.get(k)for k in['id','status','source_patterns','minimum_cardinality','minima','wall_seconds']}),flush=True)

def launch(case_id):
    r=subprocess.run([sys.executable,str(SRC/'run_source.py'),'case',case_id],cwd=ROOT,text=True,capture_output=True,timeout=75)
    return dict(id=case_id,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)

def campaign():
    assert socket.gethostname()=='loki';assert not(OUT/'source-seals.json').exists();p=read(SRC/'protocol.json');start=time.perf_counter();logs=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
        futures=[pool.submit(launch,c['id'])for c in p['cases']]
        for future in concurrent.futures.as_completed(futures):
            row=future.result();logs.append(row);print(row['stdout'].strip(),flush=True);assert row['returncode']==0,row
    assert len(logs)==18
    seals={c['id']:sha(OUT/f'cases/{c["id"]}/result.json')for c in p['cases']}
    save(OUT/'source-seals.json',dict(status='all18sources_sealed_before_any_target_diagnosis',utc=dt.datetime.now(dt.timezone.utc).isoformat(),protocol_sha256=sha(SRC/'protocol.json'),source_result_sha256=seals,campaign_wall_seconds=time.perf_counter()-start,logs=logs))
    print('ALL18SOURCE_SEAL',sha(OUT/'source-seals.json'),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['case','campaign']);a.add_argument('case_id',nargs='?');args=a.parse_args()
    if args.mode=='case':source_case(args.case_id)
    else:campaign()
