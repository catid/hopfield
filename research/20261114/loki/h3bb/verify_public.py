"""Independently enumerate supplied predictions and replay every sealed query."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import tempfile
import time
import numpy as np


def graph(n,orbits):
    chosen={tuple(o) for o in orbits};F=np.zeros((n*n,n*n),np.int32)
    for a,b in itertools.combinations(range(n*n),2):
        dr=abs(a//n-b//n);dc=abs(a%n-b%n)
        F[a,b]=F[b,a]=(dr==0 or dc==0 or dr==dc or tuple(sorted((dr,dc))) in chosen)
    return F


def direct_membership(boards,orbits):
    valid=np.ones(len(boards),bool)
    for r,s in itertools.combinations(range(13),2):
        dc=abs(boards[:,r].astype(np.int16)-boards[:,s].astype(np.int16))
        for a,b in orbits:
            valid&=~(((s-r==a)&(dc==b))|((s-r==b)&(dc==a)))
    return valid


def verify(root,require_summary=True):
    start=time.perf_counter();R=Path(root);S=R/'research/20261114/loki/h3bb';P=R/'results/research/loki/h3bb'
    read=lambda path:json.loads(path.read_text());sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest();checks=[]
    def check(name,value):assert value,name;checks.append(name)
    p=read(S/'protocol.json');g=read(P/'capacity-gate.json');raw=read(P/'study-result.json');prep=read(P/'preparation.json');partition=read(R/p['archive_partition'])
    check('budgetandcomplete',g['status']==raw['status']=='complete_verified' and g['aggregate_cpu_seconds']<=120 and g['wall_seconds']<=180 and not g['monitor_expired'] and p['budget']['GPUs']==0)
    check('frozenprotocolandstageidentities',p['frozen_utc']<prep['utc']<raw['utc'] and sha(S/'protocol.json')==g['protocol_sha256']==prep['protocol_sha256'] and sha(S/'study.py')==g['source_sha256'] and sha(P/'preparation.json')==g['preparation_sha256'] and all(g[k]==raw[k] for k in raw if k!='utc'))
    for category in ['source_sha256','source_data_sha256']:
        for rel,digest in p[category].items():check(category+' '+rel,sha(R/rel)==digest)
    classes=[c for c in partition['classes'] if c['eligible_M_at_least16'] and len(c['members'])>1];ids=sorted(cid for c in classes for cid in c['members'])
    check('all17modelsandfiveclasses',p['classes']==classes and len(classes)==5 and len(ids)==17 and prep['candidate_ids']==ids)
    predictions=[];populations=[];source_populations={}
    with tempfile.TemporaryDirectory(prefix='h3bb-public-') as temp:
        temp=Path(temp);binary=temp/'literal';subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-Wall','-Wextra','-Werror',str(S/'reverse_literal.cpp'),'-o',str(binary)],check=True)
        def enumerate_F(n,F,name):
            source=temp/f'{name}-F.bin';target=temp/f'{name}-columns.bin';F.astype(np.uint8).tofile(source)
            r=subprocess.run([str(binary),str(n),str(source),str(target)],capture_output=True,text=True,check=True)
            columns=np.fromfile(target,np.uint8).reshape(-1,n);assert len(columns)==int(r.stdout);return columns
        boards=enumerate_F(13,graph(13,[]),'ordinary')
        check('completeordinaryuniverselexorder',boards.shape==(73712,13) and np.array_equal(boards,np.load(P/'ordinary13.npy')) and all(bytes(a)<bytes(b) for a,b in zip(boards,boards[1:])))
        for cid in ids:
            info=partition['candidates'][cid];orbits=info['orbits'];F12=graph(12,orbits);source=enumerate_F(12,F12,f'N12c{cid}');source_populations[cid]=source
            check(f'givenN12source{cid}',len(source)==info['complete_source_cardinality'] and hashlib.sha256(source.tobytes()).hexdigest()==info['source_columns_bytes_sha256'] and hashlib.sha256(F12.tobytes()).hexdigest()==info['source_F_bytes_sha256'])
            F=graph(13,orbits);columns=enumerate_F(13,F,f'N13c{cid}');mask=direct_membership(boards,orbits);archive=R/f'results/research/loki/h3ba/N13/candidate{cid}'
            check(f'N13candidate{cid}independentliteralfilterarchive',np.array_equal(columns,boards[mask]) and np.array_equal(columns,np.load(archive/'columns.npy')) and np.array_equal(F,np.load(archive/'F.npy')))
            predictions.append(mask);populations.append(dict(candidate=cid,states=len(columns),validity_bytes_sha256=hashlib.sha256(mask.tobytes()).hexdigest()))
        for c in classes:check(f'givenN12class{c["class_id"]}',all(np.array_equal(source_populations[cid],source_populations[c['members'][0]]) for cid in c['members']))
    predictions=np.array(predictions,bool)
    check('allpreparedpredictions',np.array_equal(predictions,np.load(P/'predictions.npy')) and populations==prep['populations'] and prep['ordinary_sha256']==sha(P/'ordinary13.npy') and prep['predictions_sha256']==sha(P/'predictions.npy'))
    results=[];same_state_queries={};sealed_query_count=0
    for case,truth in enumerate(ids):
        group=next(c for c in classes if truth in c['members'])
        for method in ['active','random']:
            folder=P/f'cases/case{case}/{method}';candidates=list(group['members']);queried=[];steps=[]
            if method=='random':
                plan=read(folder/'random-plan.json');seed=2026118000+truth
                order=np.random.default_rng(seed).choice(len(boards),size=4,replace=False).tolist()
                check(f'case{case}randomplan',p['random_control']['seed_by_candidate'][str(truth)]==seed and plan['seed']==seed and plan['query_indices']==order and plan['stop_when_unique'] and len(set(order))==4 and p['frozen_utc']<plan['utc'])
            for step in range(4):
                if len(candidates)==1:break
                qfile=folder/f'q{step}-sealed.json';afile=folder/f'q{step}-answer.json';q=read(qfile);answer=read(afile);rows=[ids.index(cid) for cid in candidates]
                request=dict(predictions_path='predictions.npy',candidate_ids=ids,candidates=candidates,queried=queried)
                check(f'case{case}{method}{step}onlyallowedselectorinputs',q['request']==request)
                if method=='active':
                    counts=np.count_nonzero(predictions[rows],axis=0);squared=(len(rows)-2*counts)**2;squared[queried]=10**8
                    index=int(np.argmin(squared));best=int(abs(len(rows)-2*counts[index]));ties=int(np.count_nonzero(squared==squared[index]))
                    expected=dict(query_index=index,imbalance=best,optimal_board_count=ties,candidate_predictions={str(cid):bool(predictions[row,index]) for cid,row in zip(candidates,rows)})
                    key=(tuple(candidates),tuple(queried));check(f'case{case}{step}truthblinddecision',key not in same_state_queries or same_state_queries[key]==index);same_state_queries[key]=index
                else:
                    index=int(order[step]);expected=dict(query_index=index,candidate_predictions={str(cid):bool(predictions[row,index]) for cid,row in zip(candidates,rows)},random_plan_sha256=sha(folder/'random-plan.json'))
                    check(f'case{case}{step}randomplanbeforequery',plan['utc']<q['utc'])
                check(f'case{case}{method}{step}frozenpolicyselection',q['selection']==expected and q['board_zero_based']==boards[index].tolist() and q['method']==method and q['step']==step and index not in queried and q['protocol_sha256']==sha(S/'protocol.json'))
                label=bool(predictions[ids.index(truth),index]);remaining=[cid for cid in candidates if bool(predictions[ids.index(cid),index])==label]
                check(f'case{case}{method}{step}sealbeforelabel',p['frozen_utc']<prep['utc']<q['utc']<=answer['utc'] and q['monotonic_ns']<answer['monotonic_ns'] and answer['query_sha256']==sha(qfile))
                check(f'case{case}{method}{step}truthandallcompatiblecandidates',answer['label']==label and answer['candidates_before']==candidates and answer['candidates_after']==remaining and truth in remaining and len(remaining)>0)
                steps.append(dict(step=step,query_index=index,label=label,candidates_before=candidates,candidates_after=remaining,query_sha256=sha(qfile),answer_sha256=sha(afile)))
                candidates=remaining;queried=queried+[index];sealed_query_count+=1
            check(f'case{case}{method}noextraqueries',len(list(folder.glob('q*-sealed.json')))==len(steps) and len(list(folder.glob('q*-answer.json')))==len(steps))
            result=dict(case=case,true_candidate=truth,N12_class=group['class_id'],method=method,initial_candidates=group['members'],remaining_candidates=candidates,unique=len(candidates)==1,labels_used=len(steps),positive_labels=sum(s['label'] for s in steps),negative_labels=sum(not s['label'] for s in steps),steps=steps)
            check(f'case{case}{method}retainedresult',result==read(folder/'result.json'));results.append(result)
    active=[r for r in results if r['method']=='active'];control=[r for r in results if r['method']=='random'];decision='supported' if all(r['unique'] for r in active) else 'falsified'
    check('all34resultsandprimary',g['results']==results and g['decision']==decision and g['active_identified']==sum(r['unique'] for r in active) and g['random_identified']==sum(r['unique'] for r in control) and g['active_labels']==sum(r['labels_used'] for r in active) and g['random_labels']==sum(r['labels_used'] for r in control) and g['active_max_labels']==max(r['labels_used'] for r in active))
    if require_summary:
        s=read(P/'summary.json');check('summaryidentities',s['status']=='validated' and all(s[k]==g[k] for k in ['decision','results','active_identified','random_identified','active_labels','random_labels','active_max_labels']) and s['protocol_sha256']==sha(S/'protocol.json') and s['verifier_source_sha256']==sha(S/'verify_public.py') and s['audit_source_sha256']==sha(S/'analyze.py'))
        for rel,digest in s['public_sha256'].items():check('export '+rel,sha(P/rel)==digest)
        plot=read(P/'plot-provenance.json');check('figureprovenance',all(sha(R/rel)==digest for rel,digest in plot['input_sha256'].items()) and all(sha(P/rel)==digest for rel,digest in plot['output_sha256'].items()))
    return dict(status='passed',checks=len(checks),all17_given_source_populations_and_target_predictions_independently_enumerated=True,all_sealed_queries_independently_replayed=sealed_query_count,selector_no_true_ID_interface_verified=True,all_compatible_candidates_retained=True,elapsed_seconds=time.perf_counter()-start)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4]);print(json.dumps(verify(parser.parse_args().root)))
