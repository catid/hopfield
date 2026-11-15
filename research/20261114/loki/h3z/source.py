"""Pooled complete N8/N10 source identification; no target geometry."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import concurrent.futures
import datetime as dt
import hashlib
import itertools
import json
import math
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[4];SRC=ROOT/'research/20261114/loki/h3z';OUT=ROOT/'artifacts/research/20261114/loki/h3z';OLD=ROOT/'results/research/loki/h3y'
sys.path.insert(0,str(SRC.parent/'h3y'));import common as H
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def independent_verify(sources,universes,lines,generators,incidences,positive_flags,checked,minima,minimum):
    counts={(r,c):0 for r in range(10)for c in range(-9,10)if r or c>0}
    for n,columns in sources.items():
        for cols in columns:
            for r in range(n):
                for t in range(r+1,n):counts[t-r,int(cols[t])-int(cols[r])]+=1
    normals=[]
    for a in range(3):
        for b in range(-2,3):
            if(a==0 and b<=0)or math.gcd(a,abs(b))!=1:continue
            vals=[v for(r,c),v in counts.items()if a*r==-b*c]
            if len(vals)>=3 and all(v==0 for v in vals):normals.append((a,b))
    assert normals==lines
    expected=sorted({(min(abs(r),abs(c)),max(abs(r),abs(c)))for(r,c),v in counts.items()if v==0 and not any(a*r+b*c==0 for a,b in lines)})
    assert expected==generators
    rebuilt=[];negative_ids=[]
    for n,universe in universes.items():
        keys={tuple(map(int,row))for row in sources[n]};flags=np.array([tuple(map(int,row))in keys for row in universe]);assert np.array_equal(flags,positive_flags[n])and flags.sum()==len(sources[n])
        literal=np.array([[g in {(min(t-r,abs(int(row[t])-int(row[r]))),max(t-r,abs(int(row[t])-int(row[r]))))for r,t in itertools.combinations(range(n),2)}for g in generators]for row in universe],bool)
        assert np.array_equal(literal,incidences[n])and not literal[flags].any()
        rebuilt.append(literal[~flags]);negative_ids.extend((n,int(k))for k in np.flatnonzero(~flags))
    incidence=np.concatenate(rebuilt);covers=[H.bits(incidence[:,k])for k in range(len(generators))];allbits=(1<<len(incidence))-1
    expected_choices=[list(c)for size in range(minimum+1)for c in itertools.combinations(range(len(generators)),size)];assert len(expected_choices)==len(checked)
    found=[]
    for chosen,rec in zip(expected_choices,checked):
        assert chosen==rec['indices'];covered=0
        for k in chosen:covered|=covers[k]
        missing=allbits^covered;pair=negative_ids[(missing&-missing).bit_length()-1]if missing else None
        assert rec['rejected_negatives']==covered.bit_count()and rec['covers_all_negatives']==(missing==0)
        assert rec['first_uncovered_source_n']==(pair[0]if pair else None)and rec['first_uncovered_universe_index']==(pair[1]if pair else None)
        assert rec['first_uncovered_columns']==(universes[pair[0]][pair[1]].astype(int).tolist()if pair else None)
        if not missing:found.append(chosen);assert len(chosen)==minimum
    assert found==minima
    return dict(status='passed',literal_pooled_counts_and_candidates_equal=True,literal_pair_incidence_equal=True,checked_subsets=len(checked),all_witnesses_verified=True,all_minima=found)

def case(family):
    assert socket.gethostname()=='loki';p=read(SRC/'protocol.json');folder=OUT/f'cases/{family}';folder.mkdir(parents=True,exist_ok=True);assert not(folder/'result.json').exists()
    for group in ['source_sha256','source_data_sha256']:
        for path,h in p[group].items():assert sha(ROOT/path)==h
    start=time.perf_counter();cpu=time.process_time();result=dict(family_id=family,protocol_sha256=sha(SRC/'protocol.json'),no_target_geometry_used=True,source_sizes=[8,10])
    def expired(*args):raise TimeoutError('Frozen60second pooled-source inference and certificate envelope')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,60)
    try:
        sources={n:np.load(OLD/f'sources/{family}-n{n}.npy')for n in[8,10]};counts={(r,c):0 for r in range(10)for c in range(-9,10)if r or c>0}
        for n,columns in sources.items():
            local,_=H.counts_and_lines(columns,n)
            for o,v in local.items():counts[o]+=v
        lines=[]
        for a,b in itertools.product(range(-2,3),repeat=2):
            if math.gcd(abs(a),abs(b))!=1 or a<0 or(a==0 and b<0):continue
            members=[v for(r,c),v in counts.items()if a*r+b*c==0]
            if members.count(0)>=3 and not any(members):lines.append((a,b))
        result.update(source_patterns_by_size={str(n):len(v)for n,v in sources.items()},inferred_lines=lines)
        if not sources[10].size or(0,1)not in lines or(1,0)not in lines:
            result.update(status='ineligible_source_or_lines',source_structural_failure=True,minima=[])
        else:
            generators,orbits=H.candidate_orbits(counts,lines);universes={};positive={};incidences={};receipts={};negative_ids=[];negative_arrays=[];arrays={}
            for n,columns in sources.items():
                key=hashlib.sha256(json.dumps(dict(n=n,lines=lines),sort_keys=True).encode()).hexdigest();cache=OLD/f'cache/{key}'
                if(cache/'receipt.json').exists():
                    receipt=read(cache/'receipt.json');assert receipt['constructor_sha256']==sha(SRC.parent/'h3y/common.py')and receipt['lines']==[list(x)for x in lines]and receipt['n']==n
                    assert sha(cache/'primary.npy')==receipt['primary_sha256']and sha(cache/'independent.npy')==receipt['independent_sha256'];u=np.load(cache/'primary.npy');other=np.load(cache/'independent.npy')
                    receipts[str(n)]=dict(cache_key=key,reused=True,receipt=receipt)
                else:
                    u=H.line_enumeration(n,lines);other=H.bit_enumeration(n,lines);receipts[str(n)]=dict(cache_key=key,reused=False)
                assert np.array_equal(u,other);universes[n]=u;keys={tuple(map(int,row))for row in columns};flags=np.array([tuple(map(int,row))in keys for row in u]);assert flags.sum()==len(columns)
                positive[n]=flags;inc=np.zeros((len(u),len(orbits)),bool)
                for k,orbit in enumerate(orbits):
                    for dr,dc in orbit:
                        if dr<n and abs(dc)<n:inc[:,k]|=(u[:,dr:].astype(int)-u[:,:n-dr].astype(int)==dc).any(1)
                assert not inc[flags].any();incidences[n]=inc;negative_arrays.append(inc[~flags]);negative_ids.extend((n,int(k))for k in np.flatnonzero(~flags))
                arrays.update({f'n{n}_source_columns':columns,f'n{n}_universe_columns':u,f'n{n}_is_positive':flags,f'n{n}_incidence':inc})
            neg=np.concatenate(negative_arrays);coverbits=[H.bits(neg[:,k])for k in range(len(orbits))];full=(1<<len(neg))-1;checked=[];minima=[];minimum=None
            for size in range(len(orbits)+1):
                for chosen in itertools.combinations(range(len(orbits)),size):
                    covered=0
                    for k in chosen:covered|=coverbits[k]
                    missing=full^covered;pair=negative_ids[(missing&-missing).bit_length()-1]if missing else None
                    checked.append(dict(indices=list(chosen),cardinality=size,rejected_negatives=covered.bit_count(),covers_all_negatives=missing==0,
                        first_uncovered_source_n=pair[0]if pair else None,first_uncovered_universe_index=pair[1]if pair else None,
                        first_uncovered_columns=universes[pair[0]][pair[1]].astype(int).tolist()if pair else None))
                    if not missing:minima.append(list(chosen))
                if minima:minimum=size;break
            assert minimum is not None
            certificate=independent_verify(sources,universes,lines,generators,incidences,positive,checked,minima,minimum)
            np.savez_compressed(folder/'certificate.npz',**arrays);save(folder/'verification.json',certificate);save(folder/'checked-subsets.json',checked)
            result.update(status='source_certified',source_structural_failure=len(minima)!=1,candidate_generators=generators,candidate_orbits=orbits,
                minimum_cardinality=minimum,minima=minima,selected_generator_sets=[[generators[k]for k in chosen]for chosen in minima],unique_minimum=len(minima)==1,
                source_negatives_by_size={str(n):int((~positive[n]).sum())for n in sources},negative_count=len(neg),checked_subsets=len(checked),cache_receipts=receipts,
                certificate_sha256=sha(folder/'certificate.npz'),verification_sha256=sha(folder/'verification.json'),checked_subsets_sha256=sha(folder/'checked-subsets.json'),
                offset_domain='Offsets representable in at least one source size; N8nonrepresentable offsets contribute neither occurrences nor negative incidences atN8.')
        elapsed=time.perf_counter()-start;cpu_elapsed=time.process_time()-cpu;assert elapsed<60 and cpu_elapsed<60
        result.update(wall_seconds=elapsed,cpu_seconds=cpu_elapsed,completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    except BaseException as e:result.update(status='incomplete_timeout'if isinstance(e,TimeoutError)else'incomplete_validation',error=repr(e),wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    finally:signal.setitimer(signal.ITIMER_REAL,0)
    save(folder/'result.json',result);print(json.dumps({k:result.get(k)for k in['family_id','status','source_patterns_by_size','selected_generator_sets','wall_seconds']}),flush=True)

def launch(family):
    r=subprocess.run([sys.executable,str(Path(__file__)),'case',family],text=True,capture_output=True,timeout=75,cwd=ROOT);return dict(family=family,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
def campaign():
    p=read(SRC/'protocol.json');assert not(OUT/'source-seals.json').exists();start=time.perf_counter();logs=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
        futures=[pool.submit(launch,f)for f in p['family_ids']]
        for future in concurrent.futures.as_completed(futures):
            rec=future.result();logs.append(rec);print(rec['stdout'].strip(),flush=True);assert rec['returncode']==0,rec
    save(OUT/'source-seals.json',dict(status='all6source_results_sealed_before_reused_target_diagnosis',utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_result_sha256={f:sha(OUT/f'cases/{f}/result.json')for f in p['family_ids']},protocol_sha256=sha(SRC/'protocol.json'),campaign_wall_seconds=time.perf_counter()-start,logs=logs))
    print('ALL6SOURCE_SEAL',sha(OUT/'source-seals.json'),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['case','campaign']);a.add_argument('family',nargs='?');args=a.parse_args()
    if args.mode=='case':case(args.family)
    else:campaign()
