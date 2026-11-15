"""Source-posterior-guided queries on the fixed 108 positive-only source banks."""
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
    start=cpu();wall=time.perf_counter();p=frozen();OUT.mkdir(parents=True,exist_ok=True);assert not(OUT/'study-result.json').exists();results=[]
    def expired(*_):raise TimeoutError('H3bd120CPU180wall gate')
    signal.signal(signal.SIGUSR1,expired)
    try:
        rule=load(SRC.parent/'h3at/source.py','h3bd_rule');orbits=[tuple(o) for o in p['candidate_prior']['orbits']];choices=[x for k in [0,1,2] for x in itertools.combinations(range(55),k)];old=read(ROOT/p['normalizers']['selection_path']);ordinary=np.load(ROOT/p['normalizers']['ordinary_path']);sourcebits=np.load(ROOT/p['normalizers']['bits_path'])['bits'];sourcehit=rule.incidence(ordinary,orbits);scores=[];score_records=[]
        archive=ROOT/'results/research/loki/h3bc';oldprep=read(archive/'preparation.json');oldsummary=read(archive/'summary.json');boards=np.load(archive/'ordinary13.npy');bits=np.load(archive/'predictions-bits.npy');assert oldsummary['audit']['status']=='passed' and oldsummary['audit']['all1541N12_masks_and_N13_literal_populations_verified'];paritystart=cpu();paritywall=time.perf_counter();hit=rule.incidence(boards,orbits)
        for cid,chosen in enumerate(choices):
            source_mask=~sourcehit[:,chosen].any(1) if chosen else np.ones(len(ordinary),bool);columns=ordinary[source_mask];F12=rule.rule_graph(12,[orbits[j] for j in chosen]);M=len(columns)
            assert np.array_equal(np.packbits(source_mask,bitorder='little'),sourcebits[cid]) and old['candidates'][cid]==dict(candidate=cid,orbit_indices=list(chosen),orbits=[list(orbits[j]) for j in chosen],complete_source_cardinality=M,source_columns_bytes_sha256=hashlib.sha256(columns.tobytes()).hexdigest(),source_F_bytes_sha256=hashlib.sha256(F12.tobytes()).hexdigest())
            score=math.comb(55,len(chosen))*math.comb(M,16) if M>=16 else None;scores.append(str(score) if score is not None else None);score_records.append(dict(candidate=cid,orbit_count=len(chosen),source_population=M,integer_score_without_common3=str(score) if score is not None else None))
            mask=~hit[:,chosen].any(1) if chosen else np.ones(len(boards),bool);packed=np.packbits(mask,bitorder='little');F=rule.rule_graph(13,[orbits[j] for j in chosen]);record=dict(candidate=cid,states=int(mask.sum()),columns_bytes_sha256=hashlib.sha256(boards[mask].tobytes()).hexdigest(),F_bytes_sha256=hashlib.sha256(F.tobytes()).hexdigest(),packed_bytes_sha256=hashlib.sha256(packed.tobytes()).hexdigest())
            assert np.array_equal(packed,bits[cid]) and record==oldprep['populations'][cid]
        parity=dict(CPU_seconds=cpu()-paritystart,wall_seconds=time.perf_counter()-paritywall,all1541_graph_filters_and_metadata_exact=True,literal_enumeration_inherited_from='H3bc',original_audit=oldsummary['audit'],original_summary_sha256=sha(archive/'summary.json'))
        shutil.copyfile(archive/'predictions-bits.npy',OUT/'predictions-bits.npy');seal(OUT/'score-values.json',scores);seal(OUT/'score-table.json',dict(utc=now(),scores=score_records,protocol_sha256=sha(SRC/'protocol.json')))
        prep=dict(utc=now(),prediction_parity=parity,scores_sha256=sha(OUT/'score-values.json'),score_table_sha256=sha(OUT/'score-table.json'),protocol_sha256=sha(SRC/'protocol.json'));save(OUT/'preparation.json',prep);sourcepred=np.unpackbits(sourcebits,axis=1,count=14200,bitorder='little');lookup={bytes(row):i for i,row in enumerate(ordinary)}
        for cfg in p['cases']:
            case=cfg['case'];truth=cfg['true_candidate'];folder=OUT/f'cases/case{case}';folder.mkdir(parents=True,exist_ok=True);sample=np.load(archive/f'cases/case{case}/source-columns.npy');sampleids=[lookup[bytes(row)] for row in sample];initial=np.flatnonzero(sourcepred[:,sampleids].all(1)).tolist();assert initial==oldsummary['results'][case]['initial_candidates'] and truth in initial and all(scores[cid] is not None for cid in initial)
            candidates=list(initial);queried=[];steps=[];terminal=None
            for step in range(4):
                if len(candidates)==1:break
                request=dict(bits_path='predictions-bits.npy',scores_path='score-values.json',board_count=len(boards),candidates=candidates,queried=queried)
                proc=subprocess.run([sys.executable,str(SRC/'selector.py')],input=json.dumps(request),cwd=OUT,capture_output=True,text=True,check=True);selection=json.loads(proc.stdout);index=selection['query_index']
                selected=dict(utc=now(),monotonic_ns=time.monotonic_ns(),step=step,request=request,selection=selection,board_zero_based=boards[index].tolist() if index is not None else None,protocol_sha256=sha(SRC/'protocol.json'))
                if index is None:
                    terminal=dict(step=step,decision=selection,receipt_sha256=seal(folder/'no-informative-query.json',selected));break
                assert index not in queried;query_sha=seal(folder/f'q{step}-sealed.json',selected);label=oracle(truth,index,bits);remaining=[cid for cid in candidates if selection['candidate_predictions'][str(cid)]==label];answer=dict(utc=now(),monotonic_ns=time.monotonic_ns(),query_sha256=query_sha,label=label,candidates_before=candidates,candidates_after=remaining);answer_sha=seal(folder/f'q{step}-answer.json',answer);assert truth in remaining
                steps.append(dict(step=step,query_index=index,label=label,candidates_before=candidates,candidates_after=remaining,query_sha256=query_sha,answer_sha256=answer_sha));candidates=remaining;queried=queried+[index]
            exact=candidates==[truth];balanced=oldsummary['results'][case]['exact_true_singleton'];pair='both_pass' if exact and balanced else 'guided_only' if exact else 'balanced_only' if balanced else 'both_fail'
            result=dict(case=case,archive=cfg['archive'],family=cfg['family'],bank=cfg['bank'],seed=cfg['seed'],true_candidate=truth,initial_candidates=initial,remaining_candidates=candidates,exact_true_singleton=exact,membership_labels_used=len(steps),positive_membership_labels=sum(s['label'] for s in steps),negative_membership_labels=sum(not s['label'] for s in steps),source_positive_labels=16,total_labels=16+len(steps),balanced_exact=balanced,paired_outcome=pair,terminal_no_information=terminal,steps=steps);results.append(result);save(folder/'result.json',result);print('GUIDED',case,'labels',len(steps),'remaining',candidates,pair,flush=True)
        total=sum(r['exact_true_singleton'] for r in results);single=sum(r['exact_true_singleton'] for r in results if r['archive']=='H3ao');endpoints=dict(total_at_least81='supported' if total>=81 else 'falsified',all8_singletons='supported' if single==8 else 'falsified');overall='supported' if all(v=='supported' for v in endpoints.values()) else 'falsified'
        save(OUT/'study-result.json',dict(status='complete_verified',utc=now(),decision=overall,primary_endpoints=endpoints,secondary_all108='supported' if total==108 else 'falsified',results=results,inherited_truth_grid=p['inherited_truth_grid'],identified=total,singleton_identified=single,membership_labels=sum(r['membership_labels_used'] for r in results),paired_counts={key:sum(r['paired_outcome']==key for r in results) for key in ['both_pass','guided_only','balanced_only','both_fail']},preparation_sha256=sha(OUT/'preparation.json'),stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
    except BaseException as exc:
        save(OUT/'study-result.json',dict(status='failed_or_incomplete',utc=now(),error=repr(exc),completed_results=results,partial_results_usable=False,stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json')));raise


if __name__=='__main__':main()
