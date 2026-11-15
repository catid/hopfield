"""All-new paid online early handoff on the complete archived matched-prior cohort."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import gzip
import hashlib
import json
import resource
import signal
import subprocess
import sys
import time
import numpy as np
from common import ROOT,SRC,OUT,frozen,read,save,sha,now
from compiler import Budget
from synthetic_gate import run as synthetic_gate
from restriction import gate as restriction_gate,restrict_tree
from ledger_gate import run as ledger_gate
from ledger import Ledger
from prepare import run as prepare
from policy import OnlineCompiler


def cpu():
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN);return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime


def archive_query_metadata(p):
    # Only query metadata is decoded; archived answers never enter routing.
    raw=gzip.decompress((ROOT/p['archive']['ledger']).read_bytes());assert hashlib.sha256(raw).hexdigest()==p['archive']['raw_ledger_sha256'];index={};count=0
    for line in raw.splitlines():
        if b'"kind":"query"' not in line:continue
        event=json.loads(line);assert event['kind']=='query';data=event['data'];key=(data['case'],data['selection']['query_index']);index.setdefault(key,[]).append(dict(sequence=event['sequence'],sha256=event['sha256'],policy=data['policy'],phase=data['phase']));count+=1
    assert count==11344;return index


def main():
    start=cpu();wall=time.perf_counter();p=frozen();OUT.mkdir(parents=True,exist_ok=True);assert not(OUT/'study-result.json').exists();budget=Budget(p['budget']['max_memo_states_total'],p['budget']['max_canonical_split_attempts_total']);worker=None;results=[];calls=0;compiler=None;statuses=[dict(case=i,status='pending') for i in range(4096)]
    def expired(*_):raise TimeoutError('H3bs600CPU900wall gate; whole-study incomplete')
    signal.signal(signal.SIGUSR1,expired)
    try:
        ordinary,sourcepred,sourcepop,boards,bits,scores=prepare(p);sources=read(ROOT/p['archive']['sources']);lookup={bytes(row):i for i,row in enumerate(ordinary)};initials=[]
        for source in sources:
            sample=np.array(source['columns_zero_based'],dtype=np.uint8);indices=[lookup[bytes(row)] for row in sample];initial=np.flatnonzero(sourcepred[:,indices].all(1)).tolist();assert len(indices)==len(set(indices))==8 and sourcepred[source['candidate'],indices].all() and initial==source['initial_candidates'];initials.append(initial)
        assert len(sources)==4096;references=archive_query_metadata(p);worker=subprocess.Popen([sys.executable,str(SRC/'selector.py')],cwd=OUT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        def select(request):
            worker.stdin.write(json.dumps(request)+'\n');worker.stdin.flush();line=worker.stdout.readline();assert line,'Persistent selector exited';return json.loads(line)
        ledger_gate(p,select);toy=synthetic_gate(budget);restriction=restriction_gate(toy['fixtures']);save(OUT/'restriction-gate.json',dict(utc=now(),**restriction));prep=dict(utc=now(),protocol_sha256=sha(SRC/'protocol.json'),population_parity_sha256=sha(OUT/'population-parity.json'),ledger_gate_sha256=sha(OUT/'ledger-gate.json'),synthetic_gate_sha256=sha(OUT/'synthetic-gate.json'),restriction_gate_sha256=sha(OUT/'restriction-gate.json'),preflight_sha256=sha(SRC/'preflight.json'),archived_sources_sha256=sha(ROOT/p['archive']['sources']),archived_query_metadata_records=11344,archived_answers_decoded_for_routing=0,all4096_source_filters_reconstructed_before_new_labels=True)
        save(OUT/'preparation.json',prep);predictions=np.unpackbits(bits,axis=1,count=len(boards),bitorder='little').astype(bool);compiler=OnlineCompiler(predictions,scores,sha(OUT/'predictions-bits.npy'),budget,p['budget']['max_integer_bits']);ledger=Ledger(OUT/'query-ledger.jsonl');ledger.append('study_start',dict(protocol_sha256=sha(SRC/'protocol.json'),preparation_sha256=sha(OUT/'preparation.json'),archived_ledger_raw_sha256=p['archive']['raw_ledger_sha256'],archived_source_banks_sha256=sha(ROOT/p['archive']['sources'])))
        def ask(case,truth,phase,step,candidates,queried,request,selection):
            nonlocal calls
            index=selection['query_index'];assert index is not None and index not in queried and calls<p['budget']['membership_calls'];archived=list(references.get((case,index),[]));q=ledger.append('query',dict(case=case,phase=phase,step=step,request=request,selection=selection,board_zero_based=boards[index].tolist(),archived_query_references=archived))
            label=bool((bits[truth,index//8]>>(index%8))&1);calls+=1;remaining=[c for c in candidates if selection['candidate_predictions'][str(c)]==label];a=ledger.append('answer',dict(case=case,phase=phase,step=step,query_sha256=q['sha256'],label=label,candidates_before=candidates,candidates_after=remaining));assert truth in remaining
            return remaining,dict(step=step,query_index=index,label=label,candidates_before=candidates,candidates_after=remaining,query_sequence=q['sequence'],query_sha256=q['sha256'],answer_sequence=a['sequence'],answer_sha256=a['sha256'],archived_query_references=archived)
        for source,initial in zip(sources,initials):
            case=source['case'];truth=source['candidate'];candidates=list(initial);queried=[];map_steps=[];tree_steps=[];routes=[];handoff=None;handoff_candidates=None;terminal=None
            ledger.append('archived_source_bound',dict(case=case,candidate=truth,bank=source['bank'],source_record_sha256=source['source_record_sha256'],initial_candidates=initial))
            while len(candidates)>1:
                row,proof=compiler.get(candidates);group=row['group'];assert proof['candidates']==candidates and proof['depth_limit']==4 and sha(OUT/f'group{group}/proof.json')==row['proof_sha256'];bound=ledger.append('online_compiled_tree_bound',dict(case=case,MAP_calls=len(map_steps),candidates=candidates,queried=queried,group=group,feasible=proof['feasible'],proof_sha256=row['proof_sha256'],group_result_sha256=sha(OUT/f'group{group}/result.json'),cache_key=row['cache_key']));routes.append(dict(MAP_calls=len(map_steps),candidates=candidates,group=group,feasible=proof['feasible'],bound_sequence=bound['sequence'],bound_sha256=bound['sha256']))
                if proof['feasible']:
                    handoff=len(map_steps);handoff_candidates=list(candidates);key=proof['root_state']
                    while proof['states'][key]['reason']!='singleton_leaf':
                        node=proof['states'][key];index=node['chosen_split']['board_index'];request=dict(group=group,proof_sha256=row['proof_sha256'],state=key,candidates=candidates,queried=queried);selection=dict(query_index=index,candidate_predictions={str(c):bool(predictions[c,index]) for c in candidates});candidates,step=ask(case,truth,'tree',len(queried),candidates,queried,request,selection);tree_steps.append(step);queried=queried+[index];key=node['positive_state'] if step['label'] else node['negative_state']
                    break
                if len(map_steps)==4:break
                request=dict(bits_path='predictions-bits.npy',scores_path='score-values8.json',board_count=len(boards),candidates=candidates,queried=queried);selection=select(request)
                if selection['query_index'] is None:
                    event=ledger.append('no_informative_query',dict(case=case,request=request,selection=selection));terminal=dict(sequence=event['sequence'],sha256=event['sha256']);break
                candidates,step=ask(case,truth,'map',len(queried),candidates,queried,request,selection);map_steps.append(step);queried=queried+[step['query_index']]
            result=dict(case=case,candidate=truth,bank=source['bank'],source_record_sha256=source['source_record_sha256'],initial_candidates=initial,routing=routes,map_steps=map_steps,tree_steps=tree_steps,handoff_MAP_calls=handoff,handoff_candidates=handoff_candidates,terminal_no_information=terminal,remaining_candidates=candidates,exact_true_singleton=candidates==[truth],map_calls=len(map_steps),tree_calls=len(tree_steps),membership_calls=len(queried),archived_query_reacquisitions=sum(bool(r['archived_query_references']) for r in map_steps+tree_steps),new_query_boards_absent_from_archived_case=sum(not r['archived_query_references'] for r in map_steps+tree_steps))
            assert len(map_steps)<=4 and len(tree_steps)<=4 and len(queried)<=8;ledger.append('case_result',result);results.append(result);statuses[case]['status']='complete'
            if case%256==255:print('CASES',case+1,'calls',calls,'groups',len(compiler.rows),'states',budget.states,'splits',budget.splits,flush=True)
        worker.stdin.close();assert worker.wait()==0;worker=None;save(OUT/'groups.json',compiler.rows);save(OUT/'case-results.json',results);save(OUT/'case-statuses.json',statuses);ledger.append('all_new_policy_executions_complete',dict(cases=4096,actual_membership_calls=calls,case_results_sha256=sha(OUT/'case-results.json'),groups_sha256=sha(OUT/'groups.json')))
        # Archived outcomes/checkpoints are opened only after every new policy run.
        baseline=read(ROOT/p['archive']['cases']);oldcheckpoints=read(ROOT/p['archive']['checkpoints']);comparisons=[]
        for new,old,checkpoint in zip(results,baseline,oldcheckpoints):
            assert new['case']==old['case']==checkpoint['case'] and new['candidate']==old['candidate']==checkpoint['candidate'] and new['exact_true_singleton']==old['arms']['posterior']['exact_true_singleton']
            for i,step in enumerate(new['map_steps']):assert step['query_index']==checkpoint['map_steps'][i]['query_index'] and step['label']==checkpoint['map_steps'][i]['label']
            restricted=None
            if new['handoff_MAP_calls'] is not None:
                assert set(checkpoint['candidates'])<=set(new['handoff_candidates']);proof=compiler.cache[tuple(new['handoff_candidates'])][1];restricted=restrict_tree(proof,checkpoint['candidates'])
                for cid in checkpoint['candidates']:
                    node=restricted;depth=0
                    while 'candidate' not in node:node=node['positive'] if predictions[cid,node['board_index']] else node['negative'];depth+=1
                    assert node['candidate']==cid and depth<=4
            comparisons.append(dict(case=new['case'],candidate=new['candidate'],old_exact=old['arms']['posterior']['exact_true_singleton'],new_exact=new['exact_true_singleton'],old_policy_calls=old['arms']['posterior']['policy_calls'],new_policy_calls=new['membership_calls'],old_checkpoint_subset_of_handoff=new['handoff_MAP_calls'] is not None,restricted_tree=restricted))
        assert sum(r['old_policy_calls'] for r in comparisons)==10355;save(OUT/'baseline-comparison.json',comparisons);ledger.append('postexecution_baseline_evaluation',dict(baseline_case_results_sha256=sha(ROOT/p['archive']['cases']),baseline_checkpoints_sha256=sha(ROOT/p['archive']['checkpoints']),comparisons_sha256=sha(OUT/'baseline-comparison.json'),all4096_exact_flags_identical=True,all_early_feasible_later_subsets_separable=True))
        outcome=dict(decision='supported' if 100*calls<=98*10355 else 'falsified',primary=dict(left=100*calls,right=98*10355,passing_maximum=10147),cases=4096,old_policy_calls=10355,new_policy_calls=calls,new_MAP_calls=sum(r['map_calls'] for r in results),new_tree_calls=sum(r['tree_calls'] for r in results),exact_cases=sum(r['exact_true_singleton'] for r in results),all4096_exact_flags_identical_correctness_invariant=True,baseline_infeasible_cases_retained=13,new_truth_draws=0,new_source_draws=0,new_source_positive_labels=0,historical_positive_appearances_reused=32768,handoff_MAP_call_histogram={str(i):sum(r['handoff_MAP_calls']==i for r in results) for i in range(5)},cases_without_tree_handoff=sum(r['handoff_MAP_calls'] is None for r in results),archived_query_reacquisitions=sum(r['archived_query_reacquisitions'] for r in results),new_query_boards_absent_from_archived_case=sum(r['new_query_boards_absent_from_archived_case'] for r in results),distinct_compiled_sets=len(compiler.rows),distinct_feasible_sets=sum(r['feasible'] for r in compiler.rows),global_memo_states=budget.states,global_canonical_split_attempts=budget.splits)
        assert outcome['exact_cases']==4083 and calls==outcome['new_MAP_calls']+outcome['new_tree_calls'];ledger.append('study_complete',outcome);ledger.close();save(OUT/'study-result.json',dict(status='complete_verified',utc=now(),**outcome,ledger_sha256=sha(OUT/'query-ledger.jsonl'),ledger_records=ledger.sequence,ledger_last_record_sha256=ledger.previous,case_results_sha256=sha(OUT/'case-results.json'),groups_sha256=sha(OUT/'groups.json'),case_statuses_sha256=sha(OUT/'case-statuses.json'),baseline_comparison_sha256=sha(OUT/'baseline-comparison.json'),preparation_sha256=sha(OUT/'preparation.json'),stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
    except BaseException as exc:
        if worker is not None:worker.terminate();worker.wait()
        save(OUT/'case-statuses.json',statuses);save(OUT/'study-result.json',dict(status='failed_or_incomplete',utc=now(),error=repr(exc),completed_results=results,completed_groups=compiler.rows if compiler else [],actual_membership_calls=calls,global_memo_states=budget.states,global_canonical_split_attempts=budget.splits,partial_results_usable=False,cap_is_not_infeasibility=True,stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json')));raise


if __name__=='__main__':main()
