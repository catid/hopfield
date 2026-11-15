"""Mandatory H2bk exact H2bh prefixes, archives and H2bj fallback source gates."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import study as S
import selective_fallback as F
from fallback_confirmation_study import PP,DEST,protocol,load_models
from fallback_confirmation_job import run_job

SOURCE=S.OUT/'loop80_confirmation'


def compare_archives(new,old,protocol_hash,offset_adjust=0):
    count=records=0
    for path in sorted(new.glob('*.npz')):
        if 'hybrid' in path.name or 'subset' in path.name:continue
        if offset_adjust:
            if path.name=='outputs.npz':ref=old/'batch07.npz'
            else:ref=old/path.name.replace('batch00','batch07')
        else:ref=old/path.name
        with np.load(path) as z:actual={k:z[k] for k in z.files}
        with np.load(ref) as z:expected={k:z[k] for k in z.files}
        assert set(actual)==set(expected)
        for key,value in actual.items():
            if key=='trace_utf8':
                a=json.loads(value.tobytes());e=json.loads(expected[key].tobytes());assert a['protocol_sha256']==protocol_hash
                e['protocol_sha256']=protocol_hash;e['offset']-=offset_adjust;assert a==e
                records+=sum(len(r['candidates']) for r in a['trace'])
            else:assert np.array_equal(value,expected[key],equal_nan=True),(path.name,key)
        count+=1
    return {'scientific_npz_payloads_compared':count,'candidate_records_compared':records}


def main(gpu):
    started=time.perf_counter();lock,inventory=F.acquire(gpu);p=protocol();torch,_,_=S.imports()
    for path,h in p['source_gate_files'].items():assert S.sha(S.ROOT/path)==h,path
    source_ds=json.loads((SOURCE/'datasets.json').read_text())['datasets'];source_p=json.loads((SOURCE/'protocol.json').read_text());source_p['datasets']=source_ds;rows=[]
    state={'status':'running','gpu':gpu,'inventory':inventory,'completed_seeds':[],'watchdog_checks':0}
    def watch():
        import os,resource
        state.update(pid=os.getpid(),seconds=time.perf_counter()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);state['watchdog_checks']+=1
        S.save(DEST/f'gate-worker{gpu}.json',state)
        assert state['seconds']<=120 and state['peak_rss_kib']<=8*1024**2,'Frozen source gate120s/8GiB envelope'
    watch()
    for seed in ([0,2] if gpu==0 else [1]):
        models,doubles,_=load_models(p,seed);seedrow={'seed':seed}
        for shard,n,offset in [(0,256,0),(15,32,224)]:
            ds=source_ds[shard];assert S.sha(S.ROOT/ds['path'])==ds['sha256']
            with np.load(S.ROOT/ds['path']) as z:tokens=z['tokens'][offset:offset+n]
            job=next(j for j in source_p['jobs'] if j['shard']==shard and j['seed']==seed)
            target=DEST/'source-gates'/f'gpu{gpu}'/f'shard{shard:02d}-s{seed}'
            r=run_job(source_p,job,tokens,models,doubles,target,S.sha(PP),watch)
            compared=compare_archives(target,SOURCE/'runs'/job['id'],S.sha(PP),offset)
            if shard==0:
                assert r['actual_subset_inputs']==0;seedrow['all256_prefix_gate']=compared
                # Original scientific batch and final files remain byte-identical.
                assert S.sha(target/'outputs.npz')==S.sha(SOURCE/'runs'/job['id']/'outputs.npz')
                for b in range(8):assert S.sha(target/f'batch{b:02d}.npz')==S.sha(SOURCE/'runs'/job['id']/f'batch{b:02d}.npz')
            else:
                assert r['actual_subset_inputs']==1
                with np.load(target/'batch00-subset.npz') as z:sub={k:z[k] for k in z.files}
                assert sub['indices'].tolist()==[14]
                with np.load(S.OUT/'certificate_fallback/fallback'/f'i4078-s{seed}.npz') as z:
                    assert all(np.array_equal(sub[k],z[k]) for k in ['prediction','iterations','stopped','finite','final_logits'])
                with np.load(S.OUT/'certificate_fallback/diagnostics'/f'i4078-s{seed}.npz') as z:
                    assert np.array_equal(sub['own_margin'],z['margins'][0]) and np.array_equal(sub['lp_margin'],z['margins'][1])
                assert all(r['hybrid_batch_timings'][0]['subset_vs_full_control_parity'].values())
                seedrow['original32_cap_batch_and_actual_subset_gate']=compared
            watch()
        rows.append(seedrow);state['completed_seeds'].append(seed);watch()
    state.update(status='passed',seconds=time.perf_counter()-started);S.save(DEST/f'gate-worker{gpu}.json',state)
    S.save(DEST/f'source-gate-gpu{gpu}.json',{'status':'passed','protocol_sha256':S.sha(PP),'gpu':gpu,'rows':rows,'worker':state,
        'scope':'All768 H2bh shard00 model-input prefixes and96 source cap-batch model-inputs, plus all three actual H2bj subset replays. Complete original scientific arrays and traces exact, except declared protocol identity/offset metadata. Async scientific batch/final archives byte-identical.'})


def collect():
    protocol();rows=[json.loads((DEST/f'source-gate-gpu{g}.json').read_text()) for g in [0,1]]
    assert all(r['status']=='passed' and r['protocol_sha256']==S.sha(PP) and r['worker']['seconds']<=120 for r in rows)
    assert sorted(row['seed'] for r in rows for row in r['rows'])==[0,1,2]
    S.save(DEST/'numerical-gate.json',{'status':'passed','protocol_sha256':S.sha(PP),'source_shard00_model_inputs':768,
        'source_cap_batch_model_inputs':96,'actual_subset_replays':3,'rows':rows,
        'receipt_sha256':{f'source-gate-gpu{g}.json':S.sha(DEST/f'source-gate-gpu{g}.json') for g in [0,1]}})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,choices=[0,1]);a=parser.parse_args();main(a.gpu) if a.gpu is not None else collect()
