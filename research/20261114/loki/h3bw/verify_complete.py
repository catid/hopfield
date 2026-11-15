"""Independent complete all-inline policy, certificates, actual calls and baseline audit."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys
sys.set_int_max_str_digits(0)
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from audit_costs import GraphAudit,choose,exact_guard
from audit_trees import TreeAudit
from verify_tree import weighted_certificate
from verify_gates import verify as verify_serialized_gates


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def path_queries(proof,cid):
    if not proof['feasible']:return None
    i=proof['candidates'].index(cid);key=proof['root_state'];result=[]
    while proof['states'][key]['reason']!='singleton_leaf':
        node=proof['states'][key];result.append(node['chosen_split']['board_index']);key=node['positive_state'] if node['chosen_split']['positive_mask']&(1<<i) else node['negative_state']
    return result


def verify(root,archive_path=None,require_summary=False):
    started=time.perf_counter();R=Path(root).resolve();S=R/'research/20261114/loki/h3bw';P=Path(archive_path).resolve() if archive_path is not None else R/'results/research/loki/h3bw';B=R/'results/research/loki/h3bc';adapter=load(S.parent/'archive_export_layer.py','h3bw_public_archive');archive=adapter.PublicArchive(P);hashes={};proof_index={};proof_seen=set();zip_seen=set();checks=0;canonical=lambda value:json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    def read(path):
        data=archive.logical_bytes(path);digest=hashlib.sha256(data).hexdigest();hashes[str(path)]=digest
        try:relative=str(Path(path).resolve().relative_to(P))
        except ValueError:relative=None
        if relative in proof_index:
            item=proof_index[relative];packed=(P/item['compressed_path']).read_bytes();check('proof_raw_and_compressed_identity',len(data)==item['raw_bytes'] and digest==item['raw_sha256'] and len(packed)==item['compressed_bytes'] and hashlib.sha256(packed).hexdigest()==item['compressed_sha256'] and not Path(path).exists());proof_seen.add(relative)
        if relative in archive.entries:zip_seen.add(relative)
        return json.loads(data)
    def sha(path):
        key=str(path)
        if key not in hashes:hashes[key]=archive.logical_sha256(path)
        return hashes[key]
    def check(name,value):
        nonlocal checks
        assert value,name;checks+=1
    p=read(S/'protocol.json');g=read(P/'capacity-gate.json');raw=read(P/'study-result.json');prep=read(P/'preparation.json');cases=read(P/'case-results.json');groups=read(P/'groups.json');cost_registry=read(P/'counterfactual-groups.json');statuses=read(P/'case-statuses.json');source_banks=read(R/p['archive']['sources']);oldcases=read(R/p['archive']['cases']);chain=load(R/'research/20261114/loki/h3bj/verify_public.py','h3bw_chain').chain;reference=load(R/'research/20261114/loki/h3bl/verify_public.py','h3bw_reference_audit');ledger_bytes=gzip.decompress((P/'query-ledger.jsonl.gz').read_bytes());events=chain(ledger_bytes);export=read(P/'ledger-export.json');parent_bytes=gzip.decompress((R/p['archive']['ledger']).read_bytes());parent=chain(parent_bytes)
    check('whole_study_complete_within_budget',g['status']==raw['status']=='complete_verified' and not g['monitor_expired'] and g['global_memo_states']<=2000000 and g['global_split_evaluations']<=50000000 and g['aggregate_cpu_seconds']<=600 and g['wall_seconds']<=900 and all(g[k]==v for k,v in raw.items() if k!='utc'))
    check('complete_stage_source_seals',g['source_sha256']==sha(S/'study.py') and g['preparation_sha256']==sha(P/'preparation.json'))
    for field,name in [('case_results_sha256','case-results.json'),('groups_sha256','groups.json'),('case_statuses_sha256','case-statuses.json'),('baseline_comparison_sha256','baseline-comparison.json'),('counterfactual_registry_sha256','counterfactual-groups.json')]:check('complete_artifact_seal',g[field]==sha(P/name))
    for category in ['source_sha256','source_data_sha256']:
        for rel,digest in p[category].items():check('frozen_identity '+rel,sha(R/rel)==digest)
    check('stage_and_ledger_seals',g['protocol_sha256']==prep['protocol_sha256']==sha(S/'protocol.json') and p['frozen_utc']<prep['utc']<events[0]['utc']<raw['utc'] and hashlib.sha256(ledger_bytes).hexdigest()==export['raw_sha256'] and len(ledger_bytes)==export['raw_bytes'] and sha(P/'query-ledger.jsonl.gz')==export['gzip_sha256'] and hashlib.sha256(parent_bytes).hexdigest()==p['archive']['raw_ledger_sha256'])
    for field,name in [('population_parity_sha256','population-parity.json'),('ledger_gate_sha256','ledger-gate.json'),('synthetic_gate_sha256','synthetic-gate.json'),('restriction_gate_sha256','restriction-gate.json'),('cost_gate_sha256','cost-gate.json'),('pruning_gate_sha256','pruning-gate.json'),('cache_gate_sha256','cache-gate.json')]:check('runtime_gate_seal',prep[field]==sha(P/name))
    check('all_planned_cases_complete',len(cases)==len(statuses)==len(source_banks)==len(oldcases)==4096 and statuses==[dict(case=i,status='complete') for i in range(4096)] and [r['case'] for r in cases]==list(range(4096)) and sum(r['arms']['posterior']['policy_calls'] for r in oldcases)==10355)
    proof_manifest=read(P/'proof-export.json');proof_index.update({item['logical_path']:item for item in proof_manifest['proofs']});zip_manifest=archive.manifest;zip_path=P/'counterfactual-json.zip';check('lossless_export_index',proof_manifest['format']=='lossless-proof-gzip-v1' and proof_manifest['gzip_level']==6 and proof_manifest['gzip_mtime']==0 and proof_manifest['all_export_roundtrips_exact'] and len(proof_index)==len(proof_manifest['proofs'])==proof_manifest['proof_files'] and sum(item['raw_bytes'] for item in proof_index.values())==proof_manifest['raw_proof_bytes'] and sum(item['compressed_bytes'] for item in proof_index.values())==proof_manifest['compressed_proof_bytes'] and {str(path.relative_to(P))[:-3] for path in P.rglob('proof.json.gz')}==set(proof_index) and not any(P.rglob('proof.json')) and all(not Path(rel).is_absolute() and '..' not in Path(rel).parts and item['compressed_path']==rel+'.gz' for rel,item in proof_index.items()));check('counterfactual_archive_physical_identity',zip_manifest['format']=='lossless-counterfactual-zip-v1' and zip_manifest['archive']=='counterfactual-json.zip' and zip_manifest['archive_sha256']==sha(zip_path) and zip_manifest['archive_bytes']==zip_path.stat().st_size and zip_manifest['logical_files']==len(archive.entries) and zip_manifest['raw_bytes']==sum(item['raw_bytes'] for item in archive.entries.values()) and zip_manifest['all_export_roundtrips_exact'] and archive.archive.namelist()==sorted(archive.entries) and len(archive.archive.namelist())==len(set(archive.archive.namelist())) and all(not Path(rel).is_absolute() and '..' not in Path(rel).parts and adapter.is_counterfactual_json(rel) and not(P/rel).exists() for rel in archive.entries));provenance=read(P/'export-provenance.json');check('export_sources',provenance['export_source_sha256']==sha(S/'export.py') and provenance['adapter_source_sha256']==sha(S.parent/'archive_export_layer.py') and provenance['proof_adapter_source_sha256']==sha(S.parent/'export_layer.py') and provenance['raw_scientific_artifacts_unchanged'])
    gate_states=1;gate_splits=0;toy=read(P/'synthetic-gate.json')
    for fixture in toy['fixtures']:
        catalog=[dict(signature=s,lexfirst_board_index=i,board_count=1) for i,s in enumerate(fixture['signatures'])];proof=fixture['result'];ns,nt,_=weighted_certificate(proof,catalog,check,'weighted_gate_');check('weighted_gate_exact_cost',proof['optimal_cost']==fixture['expected_cost']);gate_states+=ns;gate_splits+=nt
    check('weighted_gate_counts',gate_states==toy['global_states_after_gate'] and gate_splits==toy['global_split_attempts_after_gate'])
    pruning=read(P/'pruning-gate.json');cache_gate=read(P/'cache-gate.json');adapter_paths={'pruning':P/'pruning-gate','cache':P/'cache-gate'}
    # The serialized gate audit accepts physical gzip ledgers and logical proof reads.
    gate_audit=verify_serialized_gates(P,S,pruning_name='pruning-gate',cache_name='cache-gate',read_json=read,hash_file=sha)
    gate_states+=gate_audit['charged_memo_states'];gate_splits+=gate_audit['charged_work_steps'];check('pruning_cache_gate_global_counts',cache_gate['global_memo_states']==gate_states and cache_gate['global_work_steps']==gate_splits)
    cost_gate=read(P/'cost-gate.json');cost_gate_states=cost_gate_splits=cost_gate_paths=0;gate_selector=gate_naive=0
    for fixture in cost_gate['fixtures']:
        folder=P/'cost-gate'/fixture['name'];n=len(fixture['scores']);pred=np.array([[bool(sig&(1<<c)) for sig in fixture['signatures']] for c in range(n)]);ids=dict(predictions_sha256=hashlib.sha256(np.packbits(pred,axis=1,bitorder='little').tobytes()).hexdigest(),scores_sha256=hashlib.sha256(canonical(fixture['scores'])).hexdigest(),selector_sha256=sha(S/'selector.py'),tree_compiler_sha256=sha(S/'compiler.py'),tree_wrapper_sha256=sha(S/'policy.py'));rows=[read(folder/f'group{i}/result.json') for i in range(fixture['tree_groups'])];trees=TreeAudit(folder,S,rows,pred,fixture['scores'],ids['predictions_sha256'],check,read,sha,canonical);select=lambda candidates,queried:choose(pred,fixture['scores'],candidates,queried);costs=GraphAudit(folder,None,pred,select,trees.get,ids,check,read,sha,canonical,archive.logical_cost_groups);naive_calls=0
        for run in fixture['runs']:
            _,tree=trees.get(list(range(n)),cost_gate,run['requested_depth']);baseline=costs.verify(run['baseline_group'],list(range(n)),[],run['remaining_MAP'],cost_gate);decision=exact_guard(tree,baseline);check('cost_gate_guard',decision==run['guard'] and run['root_result_sha256']==sha(folder/f'cost{run["baseline_group"]}/result.json') and run['calls_by_candidate']==baseline['remaining_calls_by_candidate'] and run['exact_by_candidate']==baseline['exact_identification_by_candidate'] and (not decision['baseline_admissible'] or decision['cost_comparison_passes']));naive={};exact={}
            for truth in range(n):
                active=list(range(n));queried=[];used=0
                for _ in range(run['remaining_MAP']):
                    if len(active)==1:break
                    selection=select(active,queried);index=selection['query_index'];naive_calls+=1;active=[c for c in active if bool(pred[c,index])==bool(pred[truth,index])];queried.append(index);used+=1
                if len(active)>1:
                    _,terminal=trees.get(active,cost_gate,4);used+=terminal['leaf_depths'][str(truth)] if terminal['feasible'] else 0;exact[str(truth)]=terminal['feasible']
                else:exact[str(truth)]=True
                naive[str(truth)]=used
            check('cost_gate_independent_rollouts',naive==run['calls_by_candidate'] and exact==run['exact_by_candidate'])
            if 'remaining_only' in fixture:check('six_failed_branches_retained',sum(not value for value in exact.values())==6 and decision['reason']=='baseline_inadmissible' and not decision['handoff'])
        ns,nt=trees.finish();nc,np_=costs.finish();check('cost_gate_node_counts',nc==fixture['baseline_groups'] and np_==fixture['recursive_selector_calls'] and naive_calls==fixture['naive_selector_calls']);cost_gate_states+=ns+nc;cost_gate_splits+=nt+np_;cost_gate_paths+=costs.candidate_paths;gate_selector+=np_;gate_naive+=naive_calls
    check('cost_gate_total_counts',cost_gate_states==cost_gate['compiler_and_baseline_memo_states'] and cost_gate_splits==cost_gate['compiler_and_baseline_split_evaluations'] and gate_selector==cost_gate['recursive_selector_calls']==30 and gate_naive==cost_gate['naive_selector_calls']==152)
    for rel,digest in cost_gate['artifact_sha256'].items():check('cost_fixture_identity',sha(P/'cost-gate'/rel)==digest)
    gate_states+=cost_gate_states;gate_splits+=cost_gate_splits;check('all_runtime_gate_work',gate_states==prep['runtime_gate_memo_states']==cost_gate['global_states_after_gate'] and gate_splits==prep['runtime_gate_split_evaluations']==cost_gate['global_split_evaluations_after_gate'])
    bits=np.load(B/'predictions-bits.npy');boards=np.load(B/'ordinary13.npy');pred=np.unpackbits(bits,axis=1,count=len(boards),bitorder='little').astype(bool);scores=read(P/'score-values8.json');check('scores_pinned',sha(P/'score-values8.json')==sha(R/'results/research/loki/h3bm/score-values8.json'));ordinary=np.load(R/p['normalizers']['ordinary_path']);source_bits=np.load(R/p['normalizers']['bits_path'])['bits'];source_pred=np.unpackbits(source_bits,axis=1,count=len(ordinary),bitorder='little').astype(bool);lookup={bytes(row):i for i,row in enumerate(ordinary)};initials=[]
    for source in source_banks:
        sample=np.array(source['columns_zero_based'],np.uint8);indices=[lookup[bytes(row)] for row in sample];initial=np.flatnonzero(source_pred[:,indices].all(1)).tolist();check('archived_source_filter',len(indices)==len(set(indices))==8 and source_pred[source['candidate'],indices].all() and initial==source['initial_candidates']);initials.append(initial)
    references={}
    for event in parent:
        if event['kind']=='query':
            d=event['data'];references.setdefault((d['case'],d['selection']['query_index']),[]).append(dict(sequence=event['sequence'],sha256=event['sha256'],policy=d['policy'],phase=d['phase']))
    ids=dict(predictions_sha256=sha(B/'predictions-bits.npy'),scores_sha256=sha(P/'score-values8.json'),selector_sha256=sha(S/'selector.py'),tree_compiler_sha256=sha(S/'compiler.py'),tree_wrapper_sha256=sha(S/'policy.py'));trees=TreeAudit(P,S,groups,pred,scores,ids['predictions_sha256'],check,read,sha,canonical);cache={}
    def select(candidates,queried):
        key=(tuple(candidates),tuple(queried))
        if key not in cache:cache[key]=choose(pred,scores,candidates,queried)
        return cache[key]
    costs=GraphAudit(P,cost_registry,pred,select,trees.get,ids,check,read,sha,canonical,archive.logical_cost_groups);at=0;paid=0
    def take(kind):
        nonlocal at
        e=events[at];at+=1;check('ledger_expected_kind',e['kind']==kind);return e
    check('study_start_binding',take('study_start')['data']==dict(protocol_sha256=sha(S/'protocol.json'),preparation_sha256=sha(P/'preparation.json'),archived_ledger_raw_sha256=p['archive']['raw_ledger_sha256'],archived_source_banks_sha256=sha(R/p['archive']['sources'])))
    def replay(case,truth,phase,step,candidates,queried,request,selection):
        nonlocal paid
        index=selection['query_index'];q=take('query');a=take('answer');prior=references.get((case,index),[]);label=bool(pred[truth,index]);remaining=[c for c in candidates if bool(pred[c,index])==label];check('all_new_paid_query_fields',q['data']==dict(case=case,phase=phase,step=step,request=request,selection=selection,board_zero_based=boards[index].tolist(),archived_query_references=prior) and a['data']==dict(case=case,phase=phase,step=step,query_sha256=q['sha256'],label=label,candidates_before=candidates,candidates_after=remaining) and index not in queried and truth in remaining and q['monotonic_ns']<a['monotonic_ns']);paid+=1
        return remaining,dict(step=step,query_index=index,label=label,candidates_before=candidates,candidates_after=remaining,query_sequence=q['sequence'],query_sha256=q['sha256'],answer_sequence=a['sequence'],answer_sha256=a['sha256'],archived_query_references=prior)
    for source,initial,stored in zip(source_banks,initials,cases):
        case=source['case'];truth=source['candidate'];candidates=list(initial);queried=[];maps=[];steps=[];routes=[];guards=[];handoff=None;handoff_candidates=None;handoff_depth=None;terminal=None;check('source_bound',take('archived_source_bound')['data']==dict(case=case,candidate=truth,bank=source['bank'],source_record_sha256=source['source_record_sha256'],initial_candidates=initial))
        while len(candidates)>1:
            depth=8-len(maps);proof=None
            if len(candidates)>16:
                event=take('handoff_ineligible_candidate_count');data=dict(case=case,MAP_calls=len(maps),requested_depth=depth,candidates=candidates,queried=queried,reason='candidate_count_eligibility_limit',limit=16,not_an_infeasibility_certificate=True);check('candidate_eligibility_bound',event['data']==data);routes.append(dict(data,bound_sequence=event['sequence'],bound_sha256=event['sha256']))
            else:
                event=take('online_compiled_tree_bound');row,proof=trees.get(candidates,event,depth);group=row['group'];data=dict(case=case,MAP_calls=len(maps),requested_depth=depth,candidates=candidates,queried=queried,group=group,feasible=proof['feasible'],proof_sha256=row['proof_sha256'],group_result_sha256=sha(P/f'group{group}/result.json'),cache_key=row['cache_key']);check('online_requested_depth_binding',event['data']==data);routes.append(dict(data,bound_sequence=event['sequence'],bound_sha256=event['sha256']))
            decision=None
            if proof is not None and proof['feasible']:
                event=take('posterior_cost_guard_bound');baseline=costs.verify(event['data']['baseline_cost_group'],candidates,queried,4-len(maps),event);decision=exact_guard(proof,baseline);data=dict(case=case,MAP_calls=len(maps),requested_depth=depth,tree_group=group,baseline_cost_group=baseline['group'],baseline_result_sha256=sha(P/f'cost{baseline["group"]}/result.json'),baseline_cache_key=baseline['cache_key'],**decision);check('exact_admissibility_guard',event['data']==data and (not decision['baseline_admissible'] or decision['cost_comparison_passes']));guards.append(dict(data,sequence=event['sequence'],sha256=event['sha256']))
                if len(maps)==4:check('MAP4_tie',decision['handoff'] and decision['tie'])
            if proof is not None and proof['feasible'] and decision['handoff']:
                handoff=len(maps);handoff_candidates=list(candidates);handoff_depth=depth;key=proof['root_state']
                while proof['states'][key]['reason']!='singleton_leaf':
                    node=proof['states'][key];index=node['chosen_split']['board_index'];request=dict(group=group,proof_sha256=row['proof_sha256'],state=key,candidates=candidates,queried=queried);selection=dict(query_index=index,candidate_predictions={str(c):bool(pred[c,index]) for c in candidates});candidates,step=replay(case,truth,'tree',len(queried),candidates,queried,request,selection);steps.append(step);queried=queried+[index];key=node['positive_state'] if step['label'] else node['negative_state']
                break
            if len(maps)==4:break
            request=dict(bits_path='predictions-bits.npy',scores_path='score-values8.json',board_count=len(boards),candidates=candidates,queried=queried);selection=select(candidates,queried)
            if selection['query_index'] is None:
                event=take('no_informative_query');check('no_information',event['data']==dict(case=case,request=request,selection=selection));terminal=dict(sequence=event['sequence'],sha256=event['sha256']);break
            candidates,step=replay(case,truth,'map',len(queried),candidates,queried,request,selection);maps.append(step);queried=queried+[step['query_index']]
        result=dict(case=case,candidate=truth,bank=source['bank'],source_record_sha256=source['source_record_sha256'],initial_candidates=initial,routing=routes,cost_guards=guards,map_steps=maps,tree_steps=steps,handoff_MAP_calls=handoff,handoff_candidates=handoff_candidates,handoff_depth=handoff_depth,terminal_no_information=terminal,remaining_candidates=candidates,exact_true_singleton=candidates==[truth],map_calls=len(maps),tree_calls=len(steps),membership_calls=len(queried),archived_query_reacquisitions=sum(bool(r['archived_query_references']) for r in maps+steps),new_query_boards_absent_from_archived_case=sum(not r['archived_query_references'] for r in maps+steps));check('completed_case_replay',result==stored==take('case_result')['data'] and len(maps)<=4 and len(queried)<=8 and result['exact_true_singleton']==oldcases[case]['arms']['posterior']['exact_true_singleton'])
    check('all_execution_before_archived_evaluation',take('all_new_policy_executions_complete')['data']==dict(cases=4096,actual_membership_calls=paid,case_results_sha256=sha(P/'case-results.json'),groups_sha256=sha(P/'groups.json'),counterfactual_registry_sha256=sha(P/'counterfactual-groups.json')))
    ns,nt=trees.finish();nc,np_=costs.finish();comparisons=read(P/'baseline-comparison.json');checkpoints=read(R/p['archive']['checkpoints']);baseline_calls=0;restricted_paths=0
    def restricted(proof,candidates):
        def visit(key,active):
            if len(active)==1:return dict(candidate=active[0])
            node=proof['states'][key];index=node['chosen_split']['board_index'];positive=[c for c in active if pred[c,index]];negative=[c for c in active if not pred[c,index]]
            if not positive:return visit(node['negative_state'],negative)
            if not negative:return visit(node['positive_state'],positive)
            return dict(board_index=index,negative=visit(node['negative_state'],negative),positive=visit(node['positive_state'],positive))
        return visit(proof['root_state'],candidates)
    for new,old,checkpoint,comparison in zip(cases,oldcases,checkpoints,comparisons):
        case=new['case'];truth=new['candidate'];active=list(initials[case]);queried=[]
        for step in checkpoint['map_steps']:
            selection=select(active,queried);index=selection['query_index'];label=bool(pred[truth,index]);remaining=[c for c in active if bool(pred[c,index])==label];q=parent[step['query_sequence']];a=parent[step['answer_sequence']];check('baseline_MAP_replay',q['data']['selection']==selection and q['sha256']==step['query_sha256'] and a['sha256']==step['answer_sha256'] and a['data']['label']==step['label']==label and step['candidates_before']==active and step['candidates_after']==remaining);active=remaining;queried.append(index)
        check('baseline_checkpoint',checkpoint['case']==case and checkpoint['candidate']==truth and checkpoint['candidates']==active and checkpoint['queried']==queried)
        for i,step in enumerate(new['map_steps']):check('new_common_MAP_prefix',step['query_index']==checkpoint['map_steps'][i]['query_index'] and step['label']==checkpoint['map_steps'][i]['label'])
        for step in old['arms']['posterior']['tree_steps']:
            index=step['query_index'];label=bool(pred[truth,index]);remaining=[c for c in active if bool(pred[c,index])==label];q=parent[step['query_sequence']];a=parent[step['answer_sequence']];check('baseline_tree_paid_call',q['sha256']==step['query_sha256'] and a['sha256']==step['answer_sha256'] and q['data']['policy']=='posterior' and q['data']['selection']['query_index']==index and a['data']['label']==step['label']==label and step['candidates_before']==active and step['candidates_after']==remaining and index not in queried);active=remaining;queried.append(index)
        check('baseline_cost_and_exact_flag',old['arms']['posterior']['policy_calls']==len(queried) and old['arms']['posterior']['exact_true_singleton']==(active==[truth])==new['exact_true_singleton']);baseline_calls+=len(queried);tree=None
        if new['handoff_MAP_calls'] is not None:
            check('handoff_later_subset',set(checkpoint['candidates'])<=set(new['handoff_candidates']));group=trees.lookup[(tuple(new['handoff_candidates']),new['handoff_depth'])];tree=restricted(trees.proofs[group],checkpoint['candidates'])
            for cid in checkpoint['candidates']:
                node=tree;depth=0
                while 'candidate' not in node:node=node['positive'] if pred[cid,node['board_index']] else node['negative'];depth+=1
                check('restricted_candidate_path',node['candidate']==cid and depth<=new['handoff_depth']);restricted_paths+=1
        expected=dict(case=case,candidate=truth,old_exact=old['arms']['posterior']['exact_true_singleton'],new_exact=new['exact_true_singleton'],old_policy_calls=old['arms']['posterior']['policy_calls'],new_policy_calls=new['membership_calls'],old_checkpoint_subset_of_handoff=new['handoff_MAP_calls'] is not None,handoff_tree_depth=new['handoff_depth'],restricted_tree=tree);check('postexecution_comparison',comparison==expected)
    check('baseline_postexecution_seal',baseline_calls==10355 and take('postexecution_baseline_evaluation')['data']==dict(baseline_case_results_sha256=sha(R/p['archive']['cases']),baseline_checkpoints_sha256=sha(R/p['archive']['checkpoints']),comparisons_sha256=sha(P/'baseline-comparison.json'),all4096_exact_flags_identical=True,all_accepted_guards_have_all_baseline_branches_identified=True))
    outcome=dict(decision='supported' if 100*paid<=98*10355 else 'falsified',primary=dict(left=100*paid,right=98*10355,passing_maximum=10147),cases=4096,old_policy_calls=10355,new_policy_calls=paid,new_MAP_calls=sum(r['map_calls'] for r in cases),new_tree_calls=sum(r['tree_calls'] for r in cases),exact_cases=sum(r['exact_true_singleton'] for r in cases),all4096_exact_flags_identical_correctness_invariant=True,baseline_infeasible_cases_retained=13,new_truth_draws=0,new_source_draws=0,new_source_positive_labels=0,historical_positive_appearances_reused=32768,handoff_MAP_call_histogram={str(i):sum(r['handoff_MAP_calls']==i for r in cases) for i in range(5)},cases_without_tree_handoff=sum(r['handoff_MAP_calls'] is None for r in cases),archived_query_reacquisitions=sum(r['archived_query_reacquisitions'] for r in cases),new_query_boards_absent_from_archived_case=sum(r['new_query_boards_absent_from_archived_case'] for r in cases),distinct_compiled_inputs=len(groups),distinct_feasible_inputs=sum(r['feasible'] for r in groups),compiled_depth_histogram={str(d):sum(r['depth']==d for r in groups) for d in range(4,9)},candidate_count_ineligible_checkpoints=sum(r.get('reason')=='candidate_count_eligibility_limit' for case in cases for r in case['routing']),tree_memo_states=sum(r['memo_states'] for r in groups),tree_canonical_split_attempts=sum(r['canonical_split_attempts'] for r in groups),huffman_merge_operations=sum(r['huffman_merge_operations'] for r in groups),compiler_work_steps=sum(r['work_steps'] for r in groups),canonical_computation_reuse_hits=sum(r['canonical_computation_reuse'] is not None for r in groups),logical_tree_states=sum(r['logical_tree_states'] for r in groups),logical_bound_states=sum(r['logical_bound_states'] for r in groups),logical_canonical_partitions=sum(r['logical_canonical_partitions'] for r in groups),logical_bound_merges=sum(r['logical_bound_merges'] for r in groups),logical_inline_merges=sum(r['logical_inline_merges'] for r in groups),logical_memoized_merges=sum(r['logical_memoized_merges'] for r in groups),counterfactual_memo_states=nc,counterfactual_MAP_partitions=np_,actual_cost_guard_evaluations=sum(len(r['cost_guards']) for r in cases),rejected_cost_guards=sum(not item['handoff'] for r in cases for item in r['cost_guards']),baseline_inadmissible_guards=sum(not item['baseline_admissible'] for r in cases for item in r['cost_guards']),admissible_cost_rejections=sum(item['baseline_admissible'] and not item['cost_comparison_passes'] for r in cases for item in r['cost_guards']),accepted_cost_guards=sum(item['handoff'] for r in cases for item in r['cost_guards']),accepted_tied_cost_guards=sum(item['tie'] for r in cases for item in r['cost_guards'] if item['handoff']),all_accepted_guards_expected_nonincreasing=True,runtime_gate_memo_states=gate_states,runtime_gate_split_evaluations=gate_splits,global_memo_states=gate_states+ns+nc,global_split_evaluations=gate_splits+nt+np_)
    check('all_primary_and_work_counts',take('study_complete')['data']==outcome and all(g[k]==v for k,v in outcome.items()) and at==len(events) and outcome['exact_cases']==4083 and outcome['admissible_cost_rejections']==0 and outcome['tree_memo_states']==ns and outcome['compiler_work_steps']==nt and outcome['canonical_computation_reuse_hits']==trees.reuse_hits)
    check('all_logical_artifacts_checked_once_or_more',proof_seen==set(proof_index) and zip_seen==set(archive.entries));storage=dict(status='passed',proof_files=len(proof_seen),raw_proof_bytes=proof_manifest['raw_proof_bytes'],compressed_proof_bytes=proof_manifest['compressed_proof_bytes'],counterfactual_logical_files=len(zip_seen),counterfactual_archive_bytes=zip_manifest['archive_bytes'],all_raw_and_compressed_identities_verified=True)
    if require_summary:
        summary=read(P/'summary.json');check('summary_complete_identity',summary['status']=='validated' and all(summary[k]==v for k,v in outcome.items()) and summary['protocol_sha256']==sha(S/'protocol.json') and summary['verifier_source_sha256']==sha(S/'verify_public.py'))
        for rel,digest in summary['verifier_dependency_sha256'].items():check('verifier_dependency',sha(R/rel)==digest)
        for rel,digest in summary['public_sha256'].items():check('public_manifest',sha(P/rel)==digest)
        plot=read(P/'plot-provenance.json')
        for rel,digest in plot['input_sha256'].items():check('plot_input',sha(R/rel)==digest)
        for rel,digest in plot['output_sha256'].items():check('plot_output',sha(P/rel)==digest)
    archive.close();return dict(status='passed',checks=checks,all4096_cases_and_exact_flags_verified=True,all13_failures_retained=True,actual_paid_queries=paid,archived_baseline_paid_queries=baseline_calls,completed_tree_inputs=len(groups),charged_tree_and_bound_memo_entries=ns,charged_compiler_work_steps=nt,canonical_computation_reuse_hits=trees.reuse_hits,logical_certificate_states=trees.logical_states,logical_certificate_work_steps=trees.logical_steps,candidate_tree_paths=trees.paths,counterfactual_nodes=nc,counterfactual_MAP_partitions=np_,counterfactual_candidate_cost_exact_entries=costs.candidate_paths,restricted_later_checkpoint_paths=restricted_paths,all_bound_order_merge_lex_pruning_and_reuse_identities_verified=True,all_depth_indexed_groups_and_bounds_reachable=True,all_baseline_branch_exact_flags_and_admissibility_guards_verified=True,whole_total_call_primary_verified=True,primary_decision=outcome['decision'],storage=storage,elapsed_seconds=time.perf_counter()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4]);parser.add_argument('--archive-dir',type=Path);args=parser.parse_args();print(json.dumps(verify(args.root,args.archive_dir)))
