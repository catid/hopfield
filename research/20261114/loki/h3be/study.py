"""Unchanged MAP guidance over all1511eligible truths with fresh positive banks."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib
import itertools
import json
import math
import resource
import shutil
import signal
import subprocess
import sys
import time
import numpy as np
from common import ROOT,SRC,OUT,frozen,read,save,sha,now,load


def cpu():
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN);return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime


def seal(path,record):
    assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    path.chmod(0o444);return sha(path)


def oracle(truth,index,bits):return bool((bits[truth,index//8]>>(index%8))&1)


def main():
    start=cpu();wall=time.perf_counter();p=frozen();OUT.mkdir(parents=True,exist_ok=True);assert not(OUT/'study-result.json').exists();results=[];worker=None
    def expired(*_):raise TimeoutError('H3be600CPU900wall gate')
    signal.signal(signal.SIGUSR1,expired)
    try:
        rule=load(SRC.parent/'h3at/source.py','h3be_rule');orbits=[tuple(o) for o in p['candidate_prior']['orbits']];choices=[x for k in [0,1,2] for x in itertools.combinations(range(55),k)];old=read(ROOT/p['normalizers']['selection_path']);ordinary=np.load(ROOT/p['normalizers']['ordinary_path']);sourcebits=np.load(ROOT/p['normalizers']['bits_path'])['bits'];sourcehit=rule.incidence(ordinary,orbits);scores=[];score_records=[]
        archive=ROOT/'results/research/loki/h3bc';oldprep=read(archive/'preparation.json');oldsummary=read(archive/'summary.json');boards=np.load(archive/'ordinary13.npy');bits=np.load(archive/'predictions-bits.npy');assert oldsummary['audit']['status']=='passed' and oldsummary['audit']['all1541N12_masks_and_N13_literal_populations_verified'];paritystart=cpu();paritywall=time.perf_counter();hit=rule.incidence(boards,orbits)
        for cid,chosen in enumerate(choices):
            source_mask=~sourcehit[:,chosen].any(1) if chosen else np.ones(len(ordinary),bool);columns=ordinary[source_mask];F12=rule.rule_graph(12,[orbits[j] for j in chosen]);M=len(columns)
            assert np.array_equal(np.packbits(source_mask,bitorder='little'),sourcebits[cid]) and old['candidates'][cid]==dict(candidate=cid,orbit_indices=list(chosen),orbits=[list(orbits[j]) for j in chosen],complete_source_cardinality=M,source_columns_bytes_sha256=hashlib.sha256(columns.tobytes()).hexdigest(),source_F_bytes_sha256=hashlib.sha256(F12.tobytes()).hexdigest())
            score=math.comb(55,len(chosen))*math.comb(M,16) if M>=16 else None;scores.append(str(score) if score is not None else None);score_records.append(dict(candidate=cid,orbit_count=len(chosen),source_population=M,integer_score_without_common3=str(score) if score is not None else None))
            mask=~hit[:,chosen].any(1) if chosen else np.ones(len(boards),bool);packed=np.packbits(mask,bitorder='little');F=rule.rule_graph(13,[orbits[j] for j in chosen]);record=dict(candidate=cid,states=int(mask.sum()),columns_bytes_sha256=hashlib.sha256(boards[mask].tobytes()).hexdigest(),F_bytes_sha256=hashlib.sha256(F.tobytes()).hexdigest(),packed_bytes_sha256=hashlib.sha256(packed.tobytes()).hexdigest())
            assert np.array_equal(packed,bits[cid]) and record==oldprep['populations'][cid]
        parity=dict(CPU_seconds=cpu()-paritystart,wall_seconds=time.perf_counter()-paritywall,all1541_graph_filters_and_metadata_exact=True,literal_enumeration_inherited_from='H3bc',original_audit=oldsummary['audit'],original_summary_sha256=sha(archive/'summary.json'))
        shutil.copyfile(archive/'predictions-bits.npy',OUT/'predictions-bits.npy');seal(OUT/'score-values.json',scores);seal(OUT/'score-table.json',dict(utc=now(),scores=score_records,protocol_sha256=sha(SRC/'protocol.json')))
        worker=subprocess.Popen([sys.executable,str(SRC/'selector.py')],cwd=OUT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        def select(request):
            worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();line=worker.stdout.readline();assert line,'Persistent selector exited';return json.loads(line)
        archival=[read(ROOT/rel) for rel in p['optimization']['archival_query_paths']];init=time.perf_counter();assert select(archival[0]['request'])==archival[0]['selection'];startup=time.perf_counter()-init;proof=[];t=time.perf_counter()
        for rel,row in zip(p['optimization']['archival_query_paths'],archival):
            actual=select(row['request']);assert actual==row['selection'];proof.append(dict(query_path=rel,query_sha256=sha(ROOT/rel),full_selection_identical=True))
        replay_time=time.perf_counter()-t;t=time.perf_counter()
        for row in archival[:8]:
            proc=subprocess.run([sys.executable,str(SRC.parent/'h3bd/selector.py')],input=json.dumps(row['request']),cwd=OUT,capture_output=True,text=True,check=True);assert json.loads(proc.stdout)==row['selection']
        cold8=time.perf_counter()-t;t=time.perf_counter()
        for row in archival[:8]:assert select(row['request'])==row['selection']
        warm8=time.perf_counter()-t;optimization=dict(utc=now(),archival_decisions=proof,all278_full_selection_dictionaries_identical=True,persistent_startup_plus_first_response_wall_seconds=startup,all278_warm_replay_wall_seconds=replay_time,first8_one_shot_wall_seconds=cold8,first8_warm_persistent_wall_seconds=warm8,per_case_state_supplied_entirely_by_request=True);seal(OUT/'runtime-parity.json',optimization)
        prep=dict(utc=now(),prediction_parity=parity,scores_sha256=sha(OUT/'score-values.json'),score_table_sha256=sha(OUT/'score-table.json'),runtime_parity_sha256=sha(OUT/'runtime-parity.json'),protocol_sha256=sha(SRC/'protocol.json'));save(OUT/'preparation.json',prep);sourcepred=np.unpackbits(sourcebits,axis=1,count=14200,bitorder='little');print('RUNTIME_PARITY',cold8,warm8,'startup',startup,flush=True)
        for cfg in p['truths']:
            truth=cfg['candidate'];folder=OUT/f'cases/candidate{truth}';folder.mkdir(parents=True,exist_ok=True)
            if not cfg['eligible']:
                result=dict(candidate=truth,orbit_count=cfg['orbit_count'],eligible=False,source_population=cfg['source_population'],seed=cfg['seed'],decision='ineligible_source_fewer_than16',exact_true_singleton=None,membership_labels_used=0,source_positive_labels=0,total_labels=0,remaining_candidates=None,steps=[]);results.append(result);save(folder/'result.json',result);continue
            population=ordinary[sourcepred[truth].astype(bool)];indices=np.random.default_rng(cfg['seed']).choice(len(population),16,replace=False);sample=population[indices];sample_hit=rule.incidence(sample,orbits);initial=[cid for cid,chosen in enumerate(choices) if not chosen or not sample_hit[:,chosen].any()];assert truth in initial and all(scores[cid] is not None for cid in initial)
            np.savez_compressed(folder/'draw.npz',indices=indices,columns=sample);draw=dict(utc=now(),candidate=truth,seed=cfg['seed'],source_population=len(population),source_positive_labels=16,initial_candidates=initial,draw_sha256=sha(folder/'draw.npz'),protocol_sha256=sha(SRC/'protocol.json'));seal(folder/'source-evidence.json',draw);candidates=list(initial);queried=[];steps=[];terminal=None
            for step in range(4):
                if len(candidates)==1:break
                request=dict(bits_path='predictions-bits.npy',scores_path='score-values.json',board_count=len(boards),candidates=candidates,queried=queried);selection=select(request);index=selection['query_index'];selected=dict(utc=now(),monotonic_ns=time.monotonic_ns(),step=step,request=request,selection=selection,board_zero_based=boards[index].tolist() if index is not None else None,protocol_sha256=sha(SRC/'protocol.json'))
                if index is None:
                    terminal=dict(step=step,decision=selection,receipt_sha256=seal(folder/'no-informative-query.json',selected));break
                assert index not in queried;query_sha=seal(folder/f'q{step}-sealed.json',selected);label=oracle(truth,index,bits);remaining=[cid for cid in candidates if selection['candidate_predictions'][str(cid)]==label];answer=dict(utc=now(),monotonic_ns=time.monotonic_ns(),query_sha256=query_sha,label=label,candidates_before=candidates,candidates_after=remaining);answer_sha=seal(folder/f'q{step}-answer.json',answer);assert truth in remaining
                steps.append(dict(step=step,query_index=index,label=label,candidates_before=candidates,candidates_after=remaining,query_sha256=query_sha,answer_sha256=answer_sha));candidates=remaining;queried=queried+[index]
            result=dict(candidate=truth,orbit_count=cfg['orbit_count'],eligible=True,source_population=cfg['source_population'],seed=cfg['seed'],decision='supported' if candidates==[truth] else 'falsified',initial_candidates=initial,remaining_candidates=candidates,exact_true_singleton=candidates==[truth],membership_labels_used=len(steps),positive_membership_labels=sum(x['label'] for x in steps),negative_membership_labels=sum(not x['label'] for x in steps),source_positive_labels=16,total_labels=16+len(steps),source_evidence_sha256=sha(folder/'source-evidence.json'),terminal_no_information=terminal,steps=steps);results.append(result);save(folder/'result.json',result)
            if truth%25==0 or not result['exact_true_singleton']:print('TRUTH',truth,'k',cfg['orbit_count'],'initial',len(initial),'labels',len(steps),'remaining',candidates,flush=True)
        worker.stdin.close();assert worker.wait()==0;worker=None;strata=[]
        for cfg in p['strata']:
            rows=[r for r in results if r['eligible'] and r['orbit_count']==cfg['orbit_count']];count=sum(r['exact_true_singleton'] for r in rows);strata.append(dict(cfg,identified=count,decision='supported' if count>=cfg['required_exact'] else 'falsified',membership_labels=sum(r['membership_labels_used'] for r in rows)))
        save(OUT/'study-result.json',dict(status='complete_verified',utc=now(),decision='supported' if all(row['decision']=='supported' for row in strata) else 'falsified',strata=strata,results=results,eligible_truths=sum(r['eligible'] for r in results),ineligible_truths=sum(not r['eligible'] for r in results),identified=sum(r['exact_true_singleton'] for r in results if r['eligible']),membership_labels=sum(r['membership_labels_used'] for r in results),preparation_sha256=sha(OUT/'preparation.json'),stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
    except BaseException as exc:
        if worker is not None:worker.terminate();worker.wait()
        save(OUT/'study-result.json',dict(status='failed_or_incomplete',utc=now(),error=repr(exc),completed_results=results,partial_results_usable=False,stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json')));raise


if __name__=='__main__':main()
