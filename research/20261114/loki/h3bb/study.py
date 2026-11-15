"""Sealed adaptive membership queries for all seventeen supplied N12 aliases."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib
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
    assert not path.exists()
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(record,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    path.chmod(0o444)
    return sha(path)


def oracle(truth,index,predictions,ids):
    return bool(predictions[ids.index(truth),index])


def main():
    start=cpu();wall=time.perf_counter();p=frozen();OUT.mkdir(parents=True,exist_ok=True)
    assert not(OUT/'study-result.json').exists();results=[]
    def expired(*_):raise TimeoutError('H3bb120CPU180wall gate')
    signal.signal(signal.SIGUSR1,expired)
    try:
        partition=read(ROOT/p['archive_partition']);classes=p['classes'];ids=sorted(cid for c in classes for cid in c['members'])
        assert classes==[c for c in partition['classes'] if c['eligible_M_at_least16'] and len(c['members'])>1]
        rule=load(SRC.parent/'h3at/source.py','h3bb_rule');helper=load(SRC.parent/'h3ad/generate.py','h3bb_ordinary')
        boards=helper.queens(13);assert boards.shape==(73712,13) and all(bytes(a)<bytes(b) for a,b in zip(boards,boards[1:]))
        np.save(OUT/'ordinary13.npy',boards);binary=OUT/'literal'
        subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-Wall','-Wextra','-Werror',str(SRC/'reverse_literal.cpp'),'-o',str(binary)],check=True)
        def literal(F,label):
            inp=OUT/f'{label}-F.bin';out=OUT/f'{label}-columns.bin';F.astype(np.uint8).tofile(inp)
            r=subprocess.run([str(binary),'13',str(inp),str(out)],capture_output=True,text=True,check=True)
            columns=np.fromfile(out,np.uint8).reshape(-1,13);assert len(columns)==int(r.stdout);return columns
        assert np.array_equal(boards,literal(rule.rule_graph(13,[]),'ordinary'))
        cells=13*np.arange(13)[None,:]+boards.astype(np.int64);a,b=np.triu_indices(13,1);predictions=[];population_rows=[]
        for cid in ids:
            archive=ROOT/f'results/research/loki/h3ba/N13/candidate{cid}';F=rule.rule_graph(13,partition['candidates'][cid]['orbits'])
            mask=~F[cells[:,a],cells[:,b]].any(1);columns=boards[mask]
            assert np.array_equal(F,np.load(archive/'F.npy')) and np.array_equal(columns,np.load(archive/'columns.npy')) and np.array_equal(columns,literal(F,f'candidate{cid}'))
            predictions.append(mask);population_rows.append(dict(candidate=cid,states=len(columns),validity_bytes_sha256=hashlib.sha256(mask.tobytes()).hexdigest()))
        predictions=np.array(predictions,bool);np.save(OUT/'predictions.npy',predictions)
        prep=dict(utc=now(),candidate_ids=ids,populations=population_rows,ordinary_sha256=sha(OUT/'ordinary13.npy'),predictions_sha256=sha(OUT/'predictions.npy'),protocol_sha256=sha(SRC/'protocol.json'))
        save(OUT/'preparation.json',prep);print('REVALIDATED',len(ids),'complete populations',flush=True)
        for case,truth in enumerate(ids):
            group=next(c for c in classes if truth in c['members'])
            for method in ['active','random']:
                folder=OUT/f'cases/case{case}/{method}';folder.mkdir(parents=True,exist_ok=True)
                candidates=list(group['members']);queried=[];steps=[];random_order=None
                if method=='random':
                    seed=p['random_control']['seed_by_candidate'][str(truth)]
                    random_order=np.random.default_rng(seed).choice(len(boards),size=4,replace=False).tolist()
                    random_seal=seal(folder/'random-plan.json',dict(utc=now(),seed=seed,query_indices=random_order,stop_when_unique=True))
                for step in range(4):
                    if len(candidates)==1:break
                    request=dict(predictions_path='predictions.npy',candidate_ids=ids,candidates=candidates,queried=queried)
                    if method=='active':
                        proc=subprocess.run([sys.executable,str(SRC/'selector.py')],input=json.dumps(request),cwd=OUT,capture_output=True,text=True,check=True)
                        selection=json.loads(proc.stdout)
                    else:
                        index=int(random_order[step]);selection=dict(query_index=index,candidate_predictions={str(cid):bool(predictions[ids.index(cid),index]) for cid in candidates},random_plan_sha256=random_seal)
                    index=selection['query_index'];assert index not in queried
                    selected=dict(utc=now(),monotonic_ns=time.monotonic_ns(),method=method,step=step,request=request,selection=selection,board_zero_based=boards[index].tolist(),protocol_sha256=sha(SRC/'protocol.json'))
                    selected_sha=seal(folder/f'q{step}-sealed.json',selected)
                    # Oracle is deliberately called only after fsync of the immutable query receipt.
                    label=oracle(truth,index,predictions,ids)
                    remaining=[cid for cid in candidates if selection['candidate_predictions'][str(cid)]==label]
                    answer=dict(utc=now(),monotonic_ns=time.monotonic_ns(),query_sha256=selected_sha,label=label,candidates_before=candidates,candidates_after=remaining)
                    answer_sha=seal(folder/f'q{step}-answer.json',answer)
                    assert truth in remaining and len(remaining)>0
                    steps.append(dict(step=step,query_index=index,label=label,candidates_before=candidates,candidates_after=remaining,query_sha256=selected_sha,answer_sha256=answer_sha))
                    candidates=remaining;queried=queried+[index]
                result=dict(case=case,true_candidate=truth,N12_class=group['class_id'],method=method,initial_candidates=group['members'],remaining_candidates=candidates,unique=len(candidates)==1,labels_used=len(steps),positive_labels=sum(s['label'] for s in steps),negative_labels=sum(not s['label'] for s in steps),steps=steps)
                results.append(result);save(folder/'result.json',result);print('QUERIES',truth,method,len(steps),candidates,flush=True)
        active=[r for r in results if r['method']=='active'];control=[r for r in results if r['method']=='random']
        save(OUT/'study-result.json',dict(status='complete_verified',utc=now(),decision='supported' if all(r['unique'] for r in active) else 'falsified',results=results,active_identified=sum(r['unique'] for r in active),random_identified=sum(r['unique'] for r in control),active_labels=sum(r['labels_used'] for r in active),random_labels=sum(r['labels_used'] for r in control),active_max_labels=max(r['labels_used'] for r in active),preparation_sha256=sha(OUT/'preparation.json'),stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
    except BaseException as exc:
        save(OUT/'study-result.json',dict(status='failed_or_incomplete',utc=now(),error=repr(exc),completed_results=results,partial_results_usable=False,stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json')));raise


if __name__=='__main__':main()
