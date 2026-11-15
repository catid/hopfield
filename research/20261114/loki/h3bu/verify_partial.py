"""Independently audit the preserved capped prefix; never compute a primary result."""
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


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def path_queries(proof,cid):
    if not proof['feasible']:return None
    i=proof['candidates'].index(cid);key=proof['root_state'];result=[]
    while proof['states'][key]['reason']!='singleton_leaf':
        node=proof['states'][key];result.append(node['chosen_split']['board_index']);key=node['positive_state'] if node['chosen_split']['positive_mask']&(1<<i) else node['negative_state']
    return result


def verify(root,archive_path=None):
    started=time.perf_counter();R=Path(root).resolve();S=R/'research/20261114/loki/h3bu';P=Path(archive_path).resolve() if archive_path is not None else R/'results/research/loki/h3bu';B=R/'results/research/loki/h3bc';adapter=load(S.parent/'archive_export_layer.py','h3bu_public_archive');archive=adapter.PublicArchive(P);hashes={};proof_index={};proof_seen=set();zip_seen=set();checks=0;canonical=lambda value:json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
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
    p=read(S/'protocol.json');g=read(P/'capacity-gate.json');raw=read(P/'study-result.json');prep=read(P/'preparation.json');cases=g['completed_results'];groups=g['completed_groups'];cost_rows=g['counterfactual_groups'];statuses=read(P/'case-statuses.json');source_banks=read(R/p['archive']['sources']);oldcases=read(R/p['archive']['cases']);chain=load(R/'research/20261114/loki/h3bj/verify_public.py','h3bu_chain').chain;reference=load(R/'research/20261114/loki/h3bl/verify_public.py','h3bu_reference_audit');ledger_bytes=gzip.decompress((P/'query-ledger.jsonl.gz').read_bytes());events=chain(ledger_bytes);export=read(P/'ledger-export.json');parent_bytes=gzip.decompress((R/p['archive']['ledger']).read_bytes());parent=chain(parent_bytes)
    check('declared_state_cap_not_hypothesis_failure',g['status']==raw['status']=='failed_or_incomplete' and g['error']=="ResourceCap('Global state cap; whole-study incomplete')" and not g['partial_results_usable'] and not g['partial_cache_usable'] and g['cap_is_not_infeasibility'] and not g['monitor_expired'] and g['global_memo_states']==p['budget']['max_memo_states_total']==2000000 and g['global_split_evaluations']<=p['budget']['max_split_evaluations_total']==50000000 and g['aggregate_cpu_seconds']<=p['budget']['aggregate_CPU_seconds']==600 and g['wall_seconds']<=p['budget']['wall_seconds']==900 and 'decision' not in g and 'primary' not in g and all(g[k]==v for k,v in raw.items() if k!='utc'))
    for category in ['source_sha256','source_data_sha256']:
        for rel,digest in p[category].items():check('frozen_identity '+rel,sha(R/rel)==digest)
    check('stage_and_ledger_seals',g['protocol_sha256']==prep['protocol_sha256']==sha(S/'protocol.json') and p['frozen_utc']<prep['utc']<events[0]['utc']<raw['utc'] and hashlib.sha256(ledger_bytes).hexdigest()==export['raw_sha256'] and len(ledger_bytes)==export['raw_bytes'] and sha(P/'query-ledger.jsonl.gz')==export['gzip_sha256'] and hashlib.sha256(parent_bytes).hexdigest()==p['archive']['raw_ledger_sha256'])
    for field,name in [('population_parity_sha256','population-parity.json'),('ledger_gate_sha256','ledger-gate.json'),('synthetic_gate_sha256','synthetic-gate.json'),('restriction_gate_sha256','restriction-gate.json'),('cost_gate_sha256','cost-gate.json'),('depth_gate_sha256','depth-gate.json')]:check('runtime_gate_seal',prep[field]==sha(P/name))
    check('all_planned_cases_retained',len(statuses)==len(source_banks)==len(oldcases)==4096 and [r['case'] for r in statuses]==list(range(4096)) and [r['case'] for r in cases]==list(range(len(cases))) and statuses==[dict(case=i,status='complete' if i<len(cases) else 'pending') for i in range(4096)] and sum(r['arms']['posterior']['policy_calls'] for r in oldcases)==10355 and all(row is not None for row in cost_rows))
    proof_manifest=read(P/'proof-export.json');proof_index.update({item['logical_path']:item for item in proof_manifest['proofs']});zip_manifest=archive.manifest;zip_path=P/'counterfactual-json.zip';check('lossless_export_index',proof_manifest['format']=='lossless-proof-gzip-v1' and proof_manifest['gzip_level']==6 and proof_manifest['gzip_mtime']==0 and proof_manifest['all_export_roundtrips_exact'] and len(proof_index)==len(proof_manifest['proofs'])==proof_manifest['proof_files'] and sum(item['raw_bytes'] for item in proof_index.values())==proof_manifest['raw_proof_bytes'] and sum(item['compressed_bytes'] for item in proof_index.values())==proof_manifest['compressed_proof_bytes'] and {str(path.relative_to(P))[:-3] for path in P.rglob('proof.json.gz')}==set(proof_index) and not any(P.rglob('proof.json')) and all(not Path(rel).is_absolute() and '..' not in Path(rel).parts and item['compressed_path']==rel+'.gz' for rel,item in proof_index.items()));check('counterfactual_archive_physical_identity',zip_manifest['format']=='lossless-counterfactual-zip-v1' and zip_manifest['archive']=='counterfactual-json.zip' and zip_manifest['archive_sha256']==sha(zip_path) and zip_manifest['archive_bytes']==zip_path.stat().st_size and zip_manifest['logical_files']==len(archive.entries) and zip_manifest['raw_bytes']==sum(item['raw_bytes'] for item in archive.entries.values()) and zip_manifest['all_export_roundtrips_exact'] and archive.archive.namelist()==sorted(archive.entries) and len(archive.archive.namelist())==len(set(archive.archive.namelist())) and all(not Path(rel).is_absolute() and '..' not in Path(rel).parts and adapter.is_counterfactual_json(rel) and not(P/rel).exists() for rel in archive.entries));provenance=read(P/'export-provenance.json');check('export_sources',provenance['export_source_sha256']==sha(S/'export.py') and provenance['adapter_source_sha256']==sha(S.parent/'archive_export_layer.py') and provenance['proof_adapter_source_sha256']==sha(S.parent/'export_layer.py') and provenance['raw_scientific_artifacts_unchanged'])
    gate_states=1;gate_splits=0;toy=read(P/'synthetic-gate.json')
    for fixture in toy['fixtures']:
        catalog=[dict(signature=s,lexfirst_board_index=i,board_count=1) for i,s in enumerate(fixture['signatures'])];proof=fixture['result'];ns,nt,_=weighted_certificate(proof,catalog,check,'weighted_gate_');check('weighted_gate_exact_cost',proof['optimal_cost']==fixture['expected_cost']);gate_states+=ns;gate_splits+=nt
    check('weighted_gate_counts',gate_states==toy['global_states_after_gate'] and gate_splits==toy['global_split_attempts_after_gate'])
    depth_gate=read(P/'depth-gate.json');ref_states=ref_splits=norm_states=norm_splits=0
    for item in depth_gate['cases']:
        catalog=[dict(signature=s,lexfirst_board_index=i,board_count=1) for i,s in enumerate(item['signatures'])];proofs={}
        for kind,checker in [('reference',reference.weighted_certificate),('normalized',weighted_certificate)]:
            proof=read(P/'depth-gate'/f'{item["name"]}-depth{item["requested_depth"]}-{kind}.json');ns,nt,_=checker(proof,catalog,check,'depth_'+kind);paths={str(c):path_queries(proof,c) for c in proof['candidates']} if proof['feasible'] else None;check('depth_gate_full_result',proof['weights']==list(map(str,item['weights'])) and proof['candidates']==item['candidates'] and proof['depth_limit']==item['requested_depth'] and item[kind]['states']==ns and item[kind]['splits']==nt and item[kind]['paths']==paths and item[kind]['feasible']==proof['feasible'] and item[kind]['optimal_cost']==proof['optimal_cost']);proofs[kind]=proof
            if kind=='reference':ref_states+=ns;ref_splits+=nt
            else:norm_states+=ns;norm_splits+=nt
        check('depth_gate_every_path_parity',proofs['reference']['feasible']==proofs['normalized']['feasible'] and proofs['reference']['optimal_cost']==proofs['normalized']['optimal_cost'] and item['reference']['paths']==item['normalized']['paths'])
    check('depth_gate_counters',len(depth_gate['cases'])==40 and depth_gate['reference_states']==ref_states and depth_gate['normalized_states']==norm_states and depth_gate['reference_splits']==ref_splits and depth_gate['normalized_splits']==norm_splits and norm_states<ref_states and norm_splits<ref_splits);gate_states+=ref_states+norm_states;gate_splits+=ref_splits+norm_splits;check('depth_gate_global',depth_gate['global_states_after_gate']==gate_states and depth_gate['global_splits_after_gate']==gate_splits)
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
    costs=GraphAudit(P,None,pred,select,trees.get,ids,check,read,sha,canonical,archive.logical_cost_groups);at=0;paid=0
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
    source=source_banks[len(cases)];initial=initials[len(cases)];check('incomplete_next_case_source_only',take('archived_source_bound')['data']==dict(case=source['case'],candidate=source['candidate'],bank=source['bank'],source_record_sha256=source['source_record_sha256'],initial_candidates=initial) and at==len(events) and paid==g['actual_membership_calls']==sum(r['membership_calls'] for r in cases) and len(cases)==1779 and paid==4402)
    ns,nt=trees.finish();nc,np_=costs.finish();check('all_counterfactual_results_exactly_preserved',nc==len(cost_rows) and all(costs.rows[i]==row for i,row in enumerate(cost_rows)));partial_states=g['global_memo_states']-gate_states-ns-nc;partial_splits=g['global_split_evaluations']-gate_splits-nt-np_;check('partial_work_not_misrepresented_as_complete_proof',partial_states>0 and partial_splits>0 and not(P/f'group{len(groups)}/result.json').exists() and not(P/f'group{len(groups)}/proof.json').exists() and not(P/f'group{len(groups)}/proof.json.gz').exists() and read(P/f'group{len(groups)}/catalog.json')['candidates']==initial)
    check('all_logical_artifacts_checked_once_or_more',proof_seen==set(proof_index) and zip_seen==set(archive.entries));storage=dict(status='passed',proof_files=len(proof_seen),raw_proof_bytes=proof_manifest['raw_proof_bytes'],compressed_proof_bytes=proof_manifest['compressed_proof_bytes'],counterfactual_logical_files=len(zip_seen),counterfactual_archive_bytes=zip_manifest['archive_bytes'],all_raw_and_compressed_identities_verified=True);archive.close();return dict(status='passed_partial_evidence_audit',scientific_status='incomplete',primary_decision=None,checks=checks,all4096_planned_statuses_retained=True,completed_cases_replayed=len(cases),pending_cases=4096-len(cases),all_actual_paid_queries_replayed=paid,all_completed_case_exact_flags_match_archive=True,all_completed_normalized_tree_certificates_verified=len(groups),completed_tree_states_verified=ns,completed_tree_splits_verified=nt,completed_tree_candidate_paths_verified=trees.paths,all_complete_counterfactual_nodes_and_exact_flags_verified=nc,counterfactual_MAP_partitions_verified=np_,counterfactual_candidate_entries_verified=costs.candidate_paths,all_completed_depth_indexed_inputs_reachable=True,all_guard_admissibility_veto_and_cost_comparisons_verified=True,runtime_gate_states_verified=gate_states,runtime_gate_splits_verified=gate_splits,interrupted_tree_states_without_complete_certificate=partial_states,interrupted_tree_splits_without_complete_certificate=partial_splits,ledger_records=len(events),ledger_sha256=hashlib.sha256(ledger_bytes).hexdigest(),storage=storage,elapsed_seconds=time.perf_counter()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4]);parser.add_argument('--archive-dir',type=Path);args=parser.parse_args();print(json.dumps(verify(args.root,args.archive_dir)))
