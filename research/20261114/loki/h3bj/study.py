"""Prospective paired policy intervention with truth-blind exact tree compilation."""
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
from compiler import Budget,compile_tree
from synthetic_gate import run as synthetic_gate
from ledger import Ledger
from ledger_gate import run as ledger_gate


def cpu():
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN);return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime


def bank_hash(sample):return hashlib.sha256(b''.join(sorted(map(bytes,sample)))).hexdigest()


def main():
    start=cpu();wall=time.perf_counter();p=frozen();OUT.mkdir(parents=True,exist_ok=True);assert not(OUT/'study-result.json').exists();results=[];worker=None;budget=Budget(p['budgets']['max_memo_states_total'],p['budgets']['max_canonical_split_attempts_total']);calls=0;groups=[];case_states=[dict(case=i,status='source_not_drawn') for i in range(1408)]
    def expired(*_):raise TimeoutError('H3bj600CPU900wall gate; whole-study incomplete')
    signal.signal(signal.SIGUSR1,expired)
    try:
        rule=load(SRC.parent/'h3at/source.py','h3bj_rule');orbits=[tuple(o) for o in p['candidate_prior']['orbits']];choices=[x for k in [0,1,2] for x in itertools.combinations(range(55),k)];old=read(ROOT/p['normalizers']['selection_path']);ordinary=np.load(ROOT/p['normalizers']['ordinary_path']);sourcebits=np.load(ROOT/p['normalizers']['bits_path'])['bits'];h12=rule.incidence(ordinary,orbits);B=ROOT/'results/research/loki/h3bc';boards=np.load(B/'ordinary13.npy');bits=np.load(B/'predictions-bits.npy');oldprep=read(B/'preparation.json');census=read(B/'summary.json');h13=rule.incidence(boards,orbits);sourcepred=[];sourcepop={};scores=[]
        assert census['audit']['all1541N12_masks_and_N13_literal_populations_verified']
        for cid,chosen in enumerate(choices):
            m12=~h12[:,chosen].any(1) if chosen else np.ones(len(ordinary),bool);columns=ordinary[m12];F12=rule.rule_graph(12,[orbits[j] for j in chosen]);assert np.array_equal(np.packbits(m12,bitorder='little'),sourcebits[cid]) and old['candidates'][cid]==dict(candidate=cid,orbit_indices=list(chosen),orbits=[list(orbits[j]) for j in chosen],complete_source_cardinality=len(columns),source_columns_bytes_sha256=hashlib.sha256(columns.tobytes()).hexdigest(),source_F_bytes_sha256=hashlib.sha256(F12.tobytes()).hexdigest());sourcepred.append(m12);sourcepop[cid]=columns;scores.append(str(math.comb(55,len(chosen))*math.comb(len(columns),16)) if len(columns)>=16 else None)
            m13=~h13[:,chosen].any(1) if chosen else np.ones(len(boards),bool);packed=np.packbits(m13,bitorder='little');F13=rule.rule_graph(13,[orbits[j] for j in chosen]);record=dict(candidate=cid,states=int(m13.sum()),columns_bytes_sha256=hashlib.sha256(boards[m13].tobytes()).hexdigest(),F_bytes_sha256=hashlib.sha256(F13.tobytes()).hexdigest(),packed_bytes_sha256=hashlib.sha256(packed.tobytes()).hexdigest());assert np.array_equal(packed,bits[cid]) and record==oldprep['populations'][cid]
        shutil.copyfile(B/'predictions-bits.npy',OUT/'predictions-bits.npy');save(OUT/'score-values.json',scores);worker=subprocess.Popen([sys.executable,str(SRC/'selector.py')],cwd=OUT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        def select(request):
            worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();line=worker.stdout.readline();assert line,'Selector exited';return json.loads(line)
        ledger_gate(p,select);synthetic_gate(budget);history={}
        def remember(key,record):history.setdefault(key,[]).append(record)
        for row in read(ROOT/'results/research/loki/h3bf/case-results.json'):remember(row['unordered_sample_sha256'],dict(study='H3bf',case=row['case'],candidate=row['candidate']))
        for row in read(ROOT/'results/research/loki/h3be/summary.json')['results']:
            if row['eligible']:
                sample=sourcepop[row['candidate']][np.random.default_rng(row['seed']).choice(row['source_population'],16,replace=False)];remember(bank_hash(sample),dict(study='H3be',candidate=row['candidate']))
        for row in census['results']:
            sample=np.load(B/f'cases/case{row["case"]}/source-columns.npy');remember(bank_hash(sample),dict(study='H3bc',case=row['case'],candidate=row['true_candidate']))
        save(OUT/'historical-banks.json',history);prep=dict(utc=now(),all1541_source_and_target_graph_parity=True,census_summary_sha256=sha(B/'summary.json'),scores_sha256=sha(OUT/'score-values.json'),historical_banks_sha256=sha(OUT/'historical-banks.json'),ledger_gate_sha256=sha(OUT/'ledger-gate.json'),synthetic_gate_sha256=sha(OUT/'synthetic-gate.json'),protocol_sha256=sha(SRC/'protocol.json'));save(OUT/'preparation.json',prep);sourcepred=np.array(sourcepred,bool);predictions=np.unpackbits(bits,axis=1,count=len(boards),bitorder='little').astype(bool);lookup={bytes(row):i for i,row in enumerate(ordinary)};ledger=Ledger(OUT/'query-ledger.jsonl');ledger.append('study_start',dict(protocol_sha256=sha(SRC/'protocol.json'),preparation_sha256=sha(OUT/'preparation.json')));sources=[];fresh_hashes={}
        # Freeze the complete source bank registry before exposing any new target label.
        for cfg in p['truths']:
            truth=cfg['candidate'];population=sourcepop[truth];assert len(population)==cfg['source_population']
            for bank,seed in enumerate(cfg['seeds']):
                case=len(sources);indices=np.random.default_rng(seed).choice(len(population),16,replace=False);sample=population[indices];initial=np.flatnonzero(sourcepred[:,[lookup[bytes(row)] for row in sample]].all(1)).tolist();key=bank_hash(sample);data=dict(case=case,candidate=truth,bank=bank,seed=seed,source_population=len(population),indices=indices.tolist(),columns_zero_based=sample.tolist(),initial_candidates=initial,unordered_sample_sha256=key,prior_same_study_cases=list(fresh_hashes.get(key,[])),historical_exact_matches=history.get(key,[]));source=ledger.append('source_draw',data);sources.append(dict(data,source_sequence=source['sequence'],source_record_sha256=source['sha256']));fresh_hashes.setdefault(key,[]).append(case);case_states[case]['status']='source_drawn_target_pending'
        save(OUT/'source-banks.json',sources);ledger.append('all_sources_drawn',dict(cases=len(sources),positive_example_appearances=22528,source_banks_sha256=sha(OUT/'source-banks.json')));cache={};all_call_sequences={}
        def ask(case,truth,arm,step,candidates,queried,request,selection):
            nonlocal calls
            index=selection['query_index'];assert index is not None and index not in queried and calls<p['budgets']['membership_calls'];key=(case,index);previous=list(all_call_sequences.get(key,[]));query=ledger.append('query',dict(case=case,arm=arm,step=step,request=request,selection=selection,board_zero_based=boards[index].tolist(),previous_same_case_query_sequences=previous));label=bool((bits[truth,index//8]>>(index%8))&1);calls+=1;remaining=[cid for cid in candidates if selection['candidate_predictions'][str(cid)]==label];answer=ledger.append('answer',dict(case=case,arm=arm,step=step,query_sha256=query['sha256'],label=label,candidates_before=candidates,candidates_after=remaining));assert truth in remaining;all_call_sequences.setdefault(key,[]).append(query['sequence']);return remaining,dict(step=step,query_index=index,label=label,candidates_before=candidates,candidates_after=remaining,query_sequence=query['sequence'],query_sha256=query['sha256'],answer_sequence=answer['sequence'],answer_sha256=answer['sha256'],previous_same_case_query_sequences=previous)
        def map_steps(case,truth,arm,candidates,queried,start_step,stop_step):
            steps=[];terminal=None
            for step in range(start_step,stop_step):
                if len(candidates)==1:break
                request=dict(bits_path='predictions-bits.npy',scores_path='score-values.json',board_count=len(boards),candidates=candidates,queried=queried);selection=select(request)
                if selection['query_index'] is None:
                    event=ledger.append('no_informative_query',dict(case=case,arm=arm,step=step,request=request,selection=selection));terminal=dict(sequence=event['sequence'],sha256=event['sha256']);break
                candidates,row=ask(case,truth,arm,step,candidates,queried,request,selection);steps.append(row);queried=queried+[row['query_index']]
            return candidates,queried,steps,terminal
        def compile_checkpoint(candidates):
            key=tuple(candidates)
            if key in cache:return cache[key]
            group=len(groups);folder=OUT/f'group{group}';folder.mkdir();before_states=budget.states;before_splits=budget.splits;start_cpu=cpu();start_wall=time.perf_counter()
            if len(candidates)>16:
                budget.state();full=(1<<len(candidates))-1;root=f'{full}:4';proof=dict(candidates=candidates,depth_limit=4,root_state=root,feasible=False,states={root:dict(mask=full,depth_remaining=4,models=len(candidates),feasible=False,reason='binary_information_bound',attempts=[])},leaf_depths={},worst_leaf_depth=None);catalog=None;save(folder/'compiler-input.json',dict(candidates=candidates,depth=4,cardinality_information_bound_only=True,allowed_inputs_only=True));catalog_sha=None
            else:
                codes=np.zeros(len(boards),np.uint32)
                for local,cid in enumerate(candidates):codes|=predictions[cid].astype(np.uint32)<<local
                signatures,first,counts=np.unique(codes,return_index=True,return_counts=True);catalog=[dict(signature=int(code),lexfirst_board_index=int(index),board_count=int(count)) for code,index,count in zip(signatures,first,counts)];save(folder/'catalog.json',dict(candidates=candidates,universe_boards=len(boards),signatures=catalog));catalog_sha=sha(folder/'catalog.json');save(folder/'compiler-input.json',dict(candidates=candidates,catalog_sha256=catalog_sha,depth=4,allowed_inputs_only=True));proof=compile_tree(candidates,catalog,4,budget)
            save(folder/'proof.json',proof)
            if proof['feasible']:
                for local,cid in enumerate(candidates):
                    node=proof['root_state'];used=[]
                    while proof['states'][node]['reason']!='singleton_leaf':
                        record=proof['states'][node];index=record['chosen_split']['board_index'];assert index not in used;used.append(index);node=record['positive_state'] if predictions[cid,index] else record['negative_state']
                    assert proof['states'][node]['mask']==1<<local and len(used)==proof['leaf_depths'][str(cid)]<=4
            row=dict(group=group,candidates=candidates,feasible=proof['feasible'],worst_leaf_depth=proof['worst_leaf_depth'],signature_count=len(catalog) if catalog is not None else None,memo_states=budget.states-before_states,canonical_split_attempts=budget.splits-before_splits,CPU_seconds=cpu()-start_cpu,wall_seconds=time.perf_counter()-start_wall,catalog_sha256=catalog_sha,compiler_input_sha256=sha(folder/'compiler-input.json'),proof_sha256=sha(folder/'proof.json'));save(folder/'result.json',row);groups.append(row);cache[key]=(row,proof);return row,proof
        for source in sources:
            case=source['case'];truth=source['candidate'];initial=source['initial_candidates'];checkpoint,queried,shared,shared_stop=map_steps(case,truth,'shared',list(initial),[],0,4);checkpoint_event=ledger.append('shared_checkpoint',dict(case=case,candidates=checkpoint,queried=queried,shared_calls=len(shared),source_record_sha256=source['source_record_sha256']));case_states[case]['status']='checkpoint_complete';group=None;proof=None
            if len(checkpoint)>1:
                group,proof=compile_checkpoint(checkpoint);decision_event=ledger.append('compiler_decision',dict(case=case,group=group['group'],result_sha256=sha(OUT/f'group{group["group"]}/result.json'),feasible=group['feasible'],checkpoint_sequence=checkpoint_event['sequence'],checkpoint_sha256=checkpoint_event['sha256']))
            else:decision_event=None
            hybrid=list(checkpoint);hybrid_queried=list(queried);hybrid_steps=[]
            if proof is not None and proof['feasible']:
                node=proof['root_state']
                while proof['states'][node]['reason']!='singleton_leaf':
                    record=proof['states'][node];index=record['chosen_split']['board_index'];request=dict(group=group['group'],proof_sha256=group['proof_sha256'],state=node,candidates=hybrid,queried=hybrid_queried);selection=dict(query_index=index,candidate_predictions={str(cid):bool(predictions[cid,index]) for cid in hybrid});hybrid,row=ask(case,truth,'hybrid',len(shared)+len(hybrid_steps),hybrid,hybrid_queried,request,selection);hybrid_steps.append(row);hybrid_queried.append(index);node=record['positive_state'] if row['label'] else record['negative_state']
                assert hybrid==[truth] and len(hybrid_steps)<=4
            baseline,base_queried,baseline_steps,base_stop=map_steps(case,truth,'baseline',list(checkpoint),list(queried),len(shared),8);shared_count=len(shared);duplicate_calls=sum(bool(r['previous_same_case_query_sequences']) for r in hybrid_steps+baseline_steps);result=dict(case=case,candidate=truth,bank=source['bank'],seed=source['seed'],source_sequence=source['source_sequence'],source_record_sha256=source['source_record_sha256'],unordered_sample_sha256=source['unordered_sample_sha256'],initial_candidates=initial,checkpoint_candidates=checkpoint,shared_checkpoint_sequence=checkpoint_event['sequence'],shared_checkpoint_sha256=checkpoint_event['sha256'],shared_steps=shared,shared_terminal_no_information=shared_stop,tree_group=group['group'] if group else None,compiler_feasible=group['feasible'] if group else None,compiler_decision_sequence=decision_event['sequence'] if decision_event else None,compiler_decision_sha256=decision_event['sha256'] if decision_event else None,hybrid_steps=hybrid_steps,hybrid_remaining_candidates=hybrid,hybrid_exact=hybrid==[truth],baseline_steps=baseline_steps,baseline_terminal_no_information=base_stop,baseline_remaining_candidates=baseline,baseline_exact=baseline==[truth],shared_membership_calls=shared_count,hybrid_additional_calls=len(hybrid_steps),baseline_additional_calls=len(baseline_steps),hybrid_policy_label_cost=shared_count+len(hybrid_steps),baseline_policy_label_cost=shared_count+len(baseline_steps),actual_membership_calls=shared_count+len(hybrid_steps)+len(baseline_steps),duplicate_cross_branch_calls=duplicate_calls,unique_same_case_query_boards=len(set(hybrid_queried+base_queried)),source_positive_appearances=16);assert result['actual_membership_calls']-duplicate_calls==result['unique_same_case_query_boards'];ledger.append('case_result',result);results.append(result);case_states[case]['status']='complete'
            if source['bank']==63:print('TRUTH',truth,'hybrid',sum(r['hybrid_exact'] for r in results[-64:]),'baseline',sum(r['baseline_exact'] for r in results[-64:]),'calls',calls,'states',budget.states,'splits',budget.splits,flush=True)
        worker.stdin.close();assert worker.wait()==0;worker=None;truth_results=[]
        for cfg in p['truths']:
            rows=[r for r in results if r['candidate']==cfg['candidate']];truth_results.append(dict(candidate=cfg['candidate'],orbits=cfg['orbits'],banks=64,hybrid_exact=sum(r['hybrid_exact'] for r in rows),baseline_exact=sum(r['baseline_exact'] for r in rows),required_exact=61,A_decision='supported' if sum(r['hybrid_exact'] for r in rows)>=61 else 'falsified',hybrid_policy_label_cost=sum(r['hybrid_policy_label_cost'] for r in rows),baseline_policy_label_cost=sum(r['baseline_policy_label_cost'] for r in rows),unique_unordered_source_banks=len({r['unordered_sample_sha256'] for r in rows})))
        hybrid_exact=sum(r['hybrid_exact'] for r in results);baseline_exact=sum(r['baseline_exact'] for r in results);unresolved=[r for r in results if len(r['checkpoint_candidates'])>1];feasible=sum(r['compiler_feasible'] for r in unresolved);outcome=dict(A_decision='supported' if all(r['A_decision']=='supported' for r in truth_results) else 'falsified',B_decision='supported' if hybrid_exact>baseline_exact else 'falsified',secondary_feasibility_decision='supported' if feasible==len(unresolved) else 'falsified',truth_results=truth_results,cases=1408,hybrid_exact=hybrid_exact,baseline_exact=baseline_exact,paired=dict(both_exact=sum(r['hybrid_exact'] and r['baseline_exact'] for r in results),hybrid_only=sum(r['hybrid_exact'] and not r['baseline_exact'] for r in results),baseline_only=sum(not r['hybrid_exact'] and r['baseline_exact'] for r in results),neither=sum(not r['hybrid_exact'] and not r['baseline_exact'] for r in results)),unresolved_checkpoints=len(unresolved),feasible_checkpoint_cases=feasible,infeasible_checkpoint_cases=len(unresolved)-feasible,distinct_compiled_sets=len(groups),shared_membership_calls=sum(r['shared_membership_calls'] for r in results),hybrid_additional_calls=sum(r['hybrid_additional_calls'] for r in results),baseline_additional_calls=sum(r['baseline_additional_calls'] for r in results),actual_membership_calls=calls,duplicate_cross_branch_calls=sum(r['duplicate_cross_branch_calls'] for r in results),unique_same_case_query_boards=sum(r['unique_same_case_query_boards'] for r in results),hybrid_policy_label_cost=sum(r['hybrid_policy_label_cost'] for r in results),baseline_policy_label_cost=sum(r['baseline_policy_label_cost'] for r in results),positive_example_appearances=22528,fresh_draw_cases_with_prior_same_study_duplicate=sum(bool(s['prior_same_study_cases']) for s in sources),fresh_draw_cases_matching_history=sum(bool(s['historical_exact_matches']) for s in sources),global_memo_states=budget.states,global_canonical_split_attempts=budget.splits)
        ledger.append('study_complete',outcome);ledger.close();save(OUT/'case-results.json',results);save(OUT/'groups.json',groups);save(OUT/'case-statuses.json',case_states);save(OUT/'study-result.json',dict(status='complete_verified',utc=now(),**outcome,ledger_sha256=sha(OUT/'query-ledger.jsonl'),ledger_records=ledger.sequence,ledger_last_record_sha256=ledger.previous,case_results_sha256=sha(OUT/'case-results.json'),groups_sha256=sha(OUT/'groups.json'),case_statuses_sha256=sha(OUT/'case-statuses.json'),source_banks_sha256=sha(OUT/'source-banks.json'),preparation_sha256=sha(OUT/'preparation.json'),stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
    except BaseException as exc:
        if worker is not None:worker.terminate();worker.wait()
        save(OUT/'case-statuses.json',case_states);save(OUT/'study-result.json',dict(status='failed_or_incomplete',utc=now(),error=repr(exc),completed_results=results,case_statuses=case_states,completed_groups=groups,actual_membership_calls=calls,global_memo_states=budget.states,global_canonical_split_attempts=budget.splits,partial_results_usable=False,cap_is_not_infeasibility=True,stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json')));raise


if __name__=='__main__':main()
