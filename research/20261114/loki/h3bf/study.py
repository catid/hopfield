"""Sixty-four new positive banks for every member of the selected failure tail."""
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
from ledger import Ledger
from ledger_gate import run as ledger_gate


def cpu():
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN);return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime


def seal(path,record):
    assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    path.chmod(0o444);return sha(path)


def oracle(truth,index,bits):return bool((bits[truth,index//8]>>(index%8))&1)


def main():
    start=cpu();wall=time.perf_counter();p=frozen();OUT.mkdir(parents=True,exist_ok=True);assert not(OUT/'study-result.json').exists();results=[];worker=None
    def expired(*_):raise TimeoutError('H3bf600CPU900wall gate')
    signal.signal(signal.SIGUSR1,expired)
    try:
        rule=load(SRC.parent/'h3at/source.py','h3bf_rule');orbits=[tuple(o) for o in p['candidate_prior']['orbits']];choices=[x for k in [0,1,2] for x in itertools.combinations(range(55),k)];old=read(ROOT/p['normalizers']['selection_path']);ordinary=np.load(ROOT/p['normalizers']['ordinary_path']);sourcebits=np.load(ROOT/p['normalizers']['bits_path'])['bits'];sourcehit=rule.incidence(ordinary,orbits);scores=[];score_records=[]
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
        gate=ledger_gate(p,select);prep=dict(utc=now(),prediction_parity=parity,scores_sha256=sha(OUT/'score-values.json'),score_table_sha256=sha(OUT/'score-table.json'),ledger_gate_sha256=sha(OUT/'ledger-gate.json'),protocol_sha256=sha(SRC/'protocol.json'));save(OUT/'preparation.json',prep);sourcepred=np.unpackbits(sourcebits,axis=1,count=14200,bitorder='little');ledger=Ledger(OUT/'query-ledger.jsonl');ledger.append('study_start',dict(protocol_sha256=sha(SRC/'protocol.json'),preparation_sha256=sha(OUT/'preparation.json')));truth_results=[]
        for cfg in p['truths']:
            truth=cfg['candidate'];population=ordinary[sourcepred[truth].astype(bool)];truth_cases=[];bank_keys=[]
            for bank,seed in enumerate(cfg['seeds']):
                case=len(results);indices=np.random.default_rng(seed).choice(len(population),16,replace=False);sample=population[indices];sample_hit=rule.incidence(sample,orbits);initial=[cid for cid,chosen in enumerate(choices) if not chosen or not sample_hit[:,chosen].any()];assert truth in initial and all(scores[cid] is not None for cid in initial);key=hashlib.sha256(b''.join(sorted(map(bytes,sample)))).hexdigest();bank_keys.append(key)
                source_record=ledger.append('source_draw',dict(case=case,candidate=truth,bank=bank,seed=seed,source_population=len(population),indices=indices.tolist(),columns_zero_based=sample.tolist(),initial_candidates=initial,unordered_sample_sha256=key));candidates=list(initial);queried=[];steps=[];terminal=None
                for step in range(4):
                    if len(candidates)==1:break
                    request=dict(bits_path='predictions-bits.npy',scores_path='score-values.json',board_count=len(boards),candidates=candidates,queried=queried);selection=select(request);index=selection['query_index'];data=dict(case=case,step=step,request=request,selection=selection,board_zero_based=boards[index].tolist() if index is not None else None)
                    if index is None:
                        receipt=ledger.append('no_informative_query',data);terminal=dict(step=step,decision=selection,record_sequence=receipt['sequence'],record_sha256=receipt['sha256']);break
                    assert index not in queried;query=ledger.append('query',data)
                    # append() has already flushed and fsynced the query before this oracle call.
                    label=oracle(truth,index,bits);remaining=[cid for cid in candidates if selection['candidate_predictions'][str(cid)]==label];answer=ledger.append('answer',dict(case=case,step=step,query_sha256=query['sha256'],label=label,candidates_before=candidates,candidates_after=remaining));assert truth in remaining
                    steps.append(dict(step=step,query_index=index,label=label,candidates_before=candidates,candidates_after=remaining,query_sequence=query['sequence'],query_sha256=query['sha256'],answer_sequence=answer['sequence'],answer_sha256=answer['sha256']));candidates=remaining;queried=queried+[index]
                result=dict(case=case,candidate=truth,bank=bank,seed=seed,source_population=len(population),initial_candidates=initial,remaining_candidates=candidates,exact_true_singleton=candidates==[truth],membership_labels_used=len(steps),positive_membership_labels=sum(x['label'] for x in steps),negative_membership_labels=sum(not x['label'] for x in steps),source_positive_labels=16,total_labels=16+len(steps),unordered_sample_sha256=key,source_sequence=source_record['sequence'],source_record_sha256=source_record['sha256'],terminal_no_information=terminal,steps=steps);ledger.append('case_result',result);results.append(result);truth_cases.append(result)
            count=sum(r['exact_true_singleton'] for r in truth_cases);row=dict(candidate=truth,orbits=cfg['orbits'],source_population=cfg['source_population'],banks=64,unique_unordered_source_banks=len(set(bank_keys)),identified=count,required_exact=48,decision='supported' if count>=48 else 'falsified',membership_labels=sum(r['membership_labels_used'] for r in truth_cases));truth_results.append(row);print('TRUTH64',truth,count,'unique_banks',row['unique_unordered_source_banks'],flush=True)
        worker.stdin.close();assert worker.wait()==0;worker=None;decision='supported' if all(r['decision']=='supported' for r in truth_results) else 'falsified';total=sum(r['exact_true_singleton'] for r in results);labels=sum(r['membership_labels_used'] for r in results);ledger.append('study_complete',dict(decision=decision,truth_results=truth_results,identified_cases=total,membership_labels=labels));ledger.close();save(OUT/'case-results.json',results)
        save(OUT/'study-result.json',dict(status='complete_verified',utc=now(),decision=decision,truth_results=truth_results,cases=len(results),identified_cases=total,membership_labels=labels,ledger_sha256=sha(OUT/'query-ledger.jsonl'),ledger_records=ledger.sequence,ledger_last_record_sha256=ledger.previous,case_results_sha256=sha(OUT/'case-results.json'),preparation_sha256=sha(OUT/'preparation.json'),stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
    except BaseException as exc:
        if worker is not None:worker.terminate();worker.wait()
        save(OUT/'study-result.json',dict(status='failed_or_incomplete',utc=now(),error=repr(exc),completed_results=results,partial_results_usable=False,stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json')));raise


if __name__=='__main__':main()
