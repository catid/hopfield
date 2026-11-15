"""Full-class positive filtering followed by sealed adaptive membership queries."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib
import itertools
import json
import resource
import signal
import subprocess
import sys
import time
import numpy as np
from common import ROOT,SRC,OUT,frozen,read,save,sha,now,load


def cpu():
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime


def seal(path,record):
    assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    path.chmod(0o444);return sha(path)


def oracle(truth,index,bits):return bool((bits[truth,index//8]>>(index%8))&1)


def main():
    start=cpu();wall=time.perf_counter();p=frozen();OUT.mkdir(parents=True,exist_ok=True);assert not(OUT/'study-result.json').exists();results=[]
    def expired(*_):raise TimeoutError('H3bc600CPU900wall gate')
    signal.signal(signal.SIGUSR1,expired)
    try:
        rule=load(SRC.parent/'h3at/source.py','h3bc_rule');helper=load(SRC.parent/'h3ad/generate.py','h3bc_ordinary');orbits=[tuple(o) for o in p['candidate_prior']['orbits']];choices=[x for k in [0,1,2] for x in itertools.combinations(range(55),k)];ids=list(range(1541))
        ordinary12=np.load(ROOT/p['normalizers']['ordinary_path']);sourcebits=np.load(ROOT/p['normalizers']['bits_path'])['bits'];old=read(ROOT/p['normalizers']['selection_path']);hit12=rule.incidence(ordinary12,orbits)
        for cid,chosen in enumerate(choices):
            mask=~hit12[:,chosen].any(1) if chosen else np.ones(len(ordinary12),bool);columns=ordinary12[mask];F=rule.rule_graph(12,[orbits[j] for j in chosen]);record=old['candidates'][cid]
            assert np.array_equal(np.packbits(mask,bitorder='little'),sourcebits[cid]) and record==dict(candidate=cid,orbit_indices=list(chosen),orbits=[list(orbits[j]) for j in chosen],complete_source_cardinality=len(columns),source_columns_bytes_sha256=hashlib.sha256(columns.tobytes()).hexdigest(),source_F_bytes_sha256=hashlib.sha256(F.tobytes()).hexdigest())
        boards=helper.queens(13);assert boards.shape==(73712,13) and all(bytes(a)<bytes(b) for a,b in zip(boards,boards[1:]));np.save(OUT/'ordinary13.npy',boards)
        binary=OUT/'literal';subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-Wall','-Wextra','-Werror',str(SRC/'reverse_literal.cpp'),'-o',str(binary)],check=True)
        def literal(F):
            inp=OUT/'scratch-F.bin';out=OUT/'scratch-columns.bin';F.astype(np.uint8).tofile(inp);r=subprocess.run([str(binary),'13',str(inp),str(out)],capture_output=True,text=True,check=True);columns=np.fromfile(out,np.uint8).reshape(-1,13);assert len(columns)==int(r.stdout);return columns
        assert np.array_equal(boards,literal(rule.rule_graph(13,[])));targetstart=cpu();targetwall=time.perf_counter();hit=rule.incidence(boards,orbits);bits=[];population_rows=[]
        for cid,chosen in enumerate(choices):
            mask=~hit[:,chosen].any(1) if chosen else np.ones(len(boards),bool);columns=boards[mask];F=rule.rule_graph(13,[orbits[j] for j in chosen]);assert np.array_equal(columns,literal(F));packed=np.packbits(mask,bitorder='little');bits.append(packed)
            population_rows.append(dict(candidate=cid,states=len(columns),columns_bytes_sha256=hashlib.sha256(columns.tobytes()).hexdigest(),F_bytes_sha256=hashlib.sha256(F.tobytes()).hexdigest(),packed_bytes_sha256=hashlib.sha256(packed.tobytes()).hexdigest()))
            if cid%100==0:print('TARGET_PARITY',cid,len(columns),flush=True)
        bits=np.array(bits,np.uint8);np.save(OUT/'predictions-bits.npy',bits);targetcost=dict(CPU_seconds=cpu()-targetstart,wall_seconds=time.perf_counter()-targetwall)
        benchmark={};outputs={}
        for mode in ['unpack_all','unpack_selected17']:
            t=time.perf_counter()
            for _ in range(2):
                expanded=np.unpackbits(bits if mode=='unpack_all' else bits[:17],axis=1,count=len(boards),bitorder='little')
                outputs[mode]=expanded[:17].copy()
            benchmark[mode+'_wall_seconds_two_repetitions']=time.perf_counter()-t
        assert np.array_equal(outputs['unpack_all'],outputs['unpack_selected17']);benchmark['exact_parity']=True;benchmark['selected_rows']=list(range(17));benchmark['all_bytes']=1541*73712;benchmark['selected_bytes']=17*73712
        prep=dict(utc=now(),candidate_ids=ids,populations=population_rows,ordinary_sha256=sha(OUT/'ordinary13.npy'),predictions_sha256=sha(OUT/'predictions-bits.npy'),target_precompute=targetcost,unpack_benchmark=benchmark,protocol_sha256=sha(SRC/'protocol.json'));save(OUT/'preparation.json',prep)
        ordinary_keys={bytes(row):i for i,row in enumerate(ordinary12)}
        for cfg in p['cases']:
            case=cfg['case'];truth=cfg['true_candidate'];folder=OUT/f'cases/case{case}';folder.mkdir(parents=True,exist_ok=True);sample=np.load(ROOT/cfg['draw_path'])[cfg['draw_key']];assert sample.shape==(16,12) and len(set(map(bytes,sample)))==16
            sampleids=[ordinary_keys[bytes(row)] for row in sample];samplehit=rule.incidence(sample,orbits);candidates=[cid for cid,chosen in enumerate(choices) if not chosen or not samplehit[:,chosen].any()];initial=list(candidates)
            sourcepred=np.unpackbits(sourcebits,axis=1,count=14200,bitorder='little');assert candidates==np.flatnonzero(sourcepred[:,sampleids].all(1)).tolist() and truth in candidates
            np.save(folder/'source-columns.npy',sample);source=dict(utc=now(),case=case,source_path=cfg['draw_path'],source_sha256=sha(ROOT/cfg['draw_path']),positive_count=16,initial_candidates=initial,source_columns_sha256=sha(folder/'source-columns.npy'),protocol_sha256=sha(SRC/'protocol.json'));seal(folder/'source-evidence.json',source);queried=[];steps=[]
            for step in range(4):
                if len(candidates)==1:break
                request=dict(bits_path='predictions-bits.npy',candidate_ids=ids,board_count=len(boards),candidates=candidates,queried=queried)
                proc=subprocess.run([sys.executable,str(SRC/'selector.py')],input=json.dumps(request),cwd=OUT,capture_output=True,text=True,check=True);selection=json.loads(proc.stdout);index=selection['query_index'];assert index not in queried
                selected=dict(utc=now(),monotonic_ns=time.monotonic_ns(),step=step,request=request,selection=selection,board_zero_based=boards[index].tolist(),protocol_sha256=sha(SRC/'protocol.json'));selected_sha=seal(folder/f'q{step}-sealed.json',selected)
                label=oracle(truth,index,bits);remaining=[cid for cid in candidates if selection['candidate_predictions'][str(cid)]==label]
                answer=dict(utc=now(),monotonic_ns=time.monotonic_ns(),query_sha256=selected_sha,label=label,candidates_before=candidates,candidates_after=remaining);answer_sha=seal(folder/f'q{step}-answer.json',answer);assert truth in remaining and len(remaining)>0
                steps.append(dict(step=step,query_index=index,label=label,candidates_before=candidates,candidates_after=remaining,query_sha256=selected_sha,answer_sha256=answer_sha));candidates=remaining;queried=queried+[index]
            result=dict(case=case,archive=cfg['archive'],family=cfg['family'],bank=cfg['bank'],seed=cfg['seed'],true_candidate=truth,initial_candidates=initial,remaining_candidates=candidates,exact_true_singleton=candidates==[truth],membership_labels_used=len(steps),positive_membership_labels=sum(x['label'] for x in steps),negative_membership_labels=sum(not x['label'] for x in steps),source_positive_labels=16,total_labels=16+len(steps),source_sha256=sha(folder/'source-evidence.json'),steps=steps)
            results.append(result);save(folder/'result.json',result);print('QUERIES',case,cfg['archive'],cfg['family'],cfg['bank'],'initial',len(initial),'labels',len(steps),'remaining',candidates,flush=True)
        save(OUT/'study-result.json',dict(status='complete_verified',utc=now(),decision='supported' if all(r['exact_true_singleton'] for r in results) else 'falsified',results=results,inherited_truth_grid=p['inherited_truth_grid'],identified=sum(r['exact_true_singleton'] for r in results),membership_labels=sum(r['membership_labels_used'] for r in results),preparation_sha256=sha(OUT/'preparation.json'),stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
    except BaseException as exc:
        save(OUT/'study-result.json',dict(status='failed_or_incomplete',utc=now(),error=repr(exc),completed_results=results,partial_results_usable=False,stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json')));raise


if __name__=='__main__':main()
