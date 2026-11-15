"""Audit inherited full populations, exact source scores, and guided decisions."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import gzip
import numpy as np


def parts(n,orbits):
    base=np.zeros((n*n,n*n),np.int32);extra=np.zeros((55,n*n,n*n),np.int32);lookup={o:j for j,o in enumerate(orbits)}
    for a,b in itertools.combinations(range(n*n),2):
        dr=abs(a//n-b//n);dc=abs(a%n-b%n)
        if dr==0 or dc==0 or dr==dc:base[a,b]=base[b,a]=1
        key=tuple(sorted((dr,dc)))
        if key in lookup:extra[lookup[key],a,b]=extra[lookup[key],b,a]=1
    return base,extra


def hits(boards,orbits):
    out=np.zeros((len(boards),55),bool)
    for i,(a,b) in enumerate(orbits):
        for r in range(boards.shape[1]):
            for s,dc in [(r+a,b),(r+b,a)]:
                if s<boards.shape[1]:out[:,i]|=abs(boards[:,r].astype(np.int16)-boards[:,s].astype(np.int16))==dc
    return out


def chain(data):
    records=[];previous='0'*64;last_mono=-1;last_utc=''
    for number,line in enumerate(data.splitlines()):
        event=json.loads(line);digest=event['sha256'];body={k:v for k,v in event.items() if k!='sha256'};canonical=json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        assert hashlib.sha256(canonical).hexdigest()==digest and event['sequence']==number and event['previous_sha256']==previous and event['monotonic_ns']>last_mono and event['utc']>=last_utc
        if event['kind']=='answer':assert records and records[-1]['kind']=='query' and event['data']['query_sha256']==records[-1]['sha256']
        records.append(event);previous=digest;last_mono=event['monotonic_ns'];last_utc=event['utc']
    assert records;return records


def verify(root,require_summary=True):
    start=time.perf_counter();R=Path(root);S=R/'research/20261114/loki/h3bh';P=R/'results/research/loki/h3bh';B=R/'results/research/loki/h3bc';F=R/'results/research/loki/h3bf';read=lambda path:json.loads(path.read_text());sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest();checks=[]
    def check(name,value):assert value,name;checks.append(name)
    p=read(S/'protocol.json');g=read(P/'capacity-gate.json');raw=read(P/'study-result.json');prep=read(P/'preparation.json');checkpoints=read(P/'continuation-checkpoints.json');old=read(R/p['normalizers']['selection_path']);census=read(B/'summary.json');oldprep=read(B/'preparation.json');prior=read(F/'summary.json');oldcases=read(F/'case-results.json');parent_bytes=gzip.decompress((F/'query-ledger.jsonl.gz').read_bytes());parent_events=chain(parent_bytes);case_events={e['data']['case']:e for e in parent_events if e['kind']=='case_result'};cases=read(P/'case-results.json');export=read(P/'ledger-export.json');ledger_bytes=gzip.decompress((P/'query-ledger.jsonl.gz').read_bytes());events=chain(ledger_bytes)
    check('wholebudgetandnewlabelcaps',g['status']==raw['status']=='complete_verified' and not g['monitor_expired'] and g['aggregate_cpu_seconds']<=120 and g['wall_seconds']<=180 and g['additional_membership_labels']<=2816 and g['new_source_draws']==0 and p['budget']['GPUs']==0)
    check('stagehashesandexactcompressedledger',p['frozen_utc']<checkpoints['utc']<prep['utc']<raw['utc'] and sha(S/'protocol.json')==g['protocol_sha256']==prep['protocol_sha256'] and sha(S/'study.py')==g['source_sha256'] and sha(P/'preparation.json')==g['preparation_sha256'] and sha(P/'continuation-checkpoints.json')==prep['checkpoints_sha256'] and sha(P/'case-results.json')==g['case_results_sha256'] and all(g[k]==raw[k] for k in raw if k!='utc') and hashlib.sha256(ledger_bytes).hexdigest()==g['ledger_sha256']==export['raw_sha256'] and export['gzip_sha256']==sha(P/'query-ledger.jsonl.gz') and len(ledger_bytes)==export['raw_bytes'] and len(events)==g['ledger_records'] and events[-1]['sha256']==g['ledger_last_record_sha256'])
    for category in ['source_sha256','source_data_sha256']:
        for rel,digest in p[category].items():check(category+' '+rel,sha(R/rel)==digest)
    check('immutableparentcensusledgerandunchangedcode',sha(S/'selector.py')==sha(R/'research/20261114/loki/h3bf/selector.py') and sha(S/'ledger.py')==sha(R/'research/20261114/loki/h3bf/ledger.py') and hashlib.sha256(parent_bytes).hexdigest()==prior['ledger_sha256']==checkpoints['parent_ledger_raw_sha256'] and len(parent_events)==prior['ledger_records']==checkpoints['parent_ledger_records'] and checkpoints['parent_summary_sha256']==sha(F/'summary.json') and checkpoints['source_sha256']==sha(S/'checkpoints.py') and prior['public_sha256']['case-results.json']==sha(F/'case-results.json') and prior['public_sha256']['query-ledger.jsonl.gz']==sha(F/'query-ledger.jsonl.gz') and prior['public_sha256']['ledger-gate.json']==sha(F/'ledger-gate.json') and prior['audit']['synthetic_prior_format_parity_and_tamper_rejection_verified'] and census['audit']['all1541N12_masks_and_N13_literal_populations_verified'] and census['public_sha256']['predictions-bits.npy']==sha(B/'predictions-bits.npy') and census['public_sha256']['preparation.json']==sha(B/'preparation.json'))
    check('same1408cases22truths',p['truths']==read(R/'research/20261114/loki/h3bf/protocol.json')['truths'] and len(oldcases)==1408 and sum(r['exact_true_singleton'] for r in oldcases)==704 and sum(r['membership_labels_used'] for r in oldcases)==5372)
    orbits=list(itertools.combinations(range(1,12),2));choices=[x for k in [0,1,2] for x in itertools.combinations(range(55),k)];ordinary=np.load(R/p['normalizers']['ordinary_path']);boards=np.load(B/'ordinary13.npy');sourcebits=np.load(R/p['normalizers']['bits_path'])['bits'];targetbits=np.load(B/'predictions-bits.npy');h12=hits(ordinary,orbits);h13=hits(boards,orbits);base12,extra12=parts(12,orbits);base13,extra13=parts(13,orbits);scores=[];score_records=[];sourcepred=[];sourcepop={}
    for cid,chosen in enumerate(choices):
        m12=~h12[:,chosen].any(1) if chosen else np.ones(len(ordinary),bool);m13=~h13[:,chosen].any(1) if chosen else np.ones(len(boards),bool);F12=base12.copy();F13=base13.copy()
        for j in chosen:F12|=extra12[j];F13|=extra13[j]
        columns=ordinary[m12];M=len(columns);sourcepop[cid]=columns;sourcepred.append(m12);record=dict(candidate=cid,orbit_indices=list(chosen),orbits=[list(orbits[j]) for j in chosen],complete_source_cardinality=M,source_columns_bytes_sha256=hashlib.sha256(columns.tobytes()).hexdigest(),source_F_bytes_sha256=hashlib.sha256(F12.tobytes()).hexdigest());check(f'candidate{cid}sourceparity',np.array_equal(np.packbits(m12,bitorder='little'),sourcebits[cid]) and record==old['candidates'][cid])
        packed=np.packbits(m13,bitorder='little');record=dict(candidate=cid,states=int(m13.sum()),columns_bytes_sha256=hashlib.sha256(boards[m13].tobytes()).hexdigest(),F_bytes_sha256=hashlib.sha256(F13.tobytes()).hexdigest(),packed_bytes_sha256=hashlib.sha256(packed.tobytes()).hexdigest());check(f'candidate{cid}targetparity',np.array_equal(packed,targetbits[cid]) and record==oldprep['populations'][cid]);score=math.comb(55,len(chosen))*math.comb(M,16) if M>=16 else None;scores.append(str(score) if score is not None else None);score_records.append(dict(candidate=cid,orbit_count=len(chosen),source_population=M,integer_score_without_common3=str(score) if score is not None else None))
    table=read(P/'score-table.json');check('scoresandcensuscoverage',table['scores']==score_records and read(P/'score-values.json')==scores and prep['scores_sha256']==sha(P/'score-values.json') and prep['score_table_sha256']==sha(P/'score-table.json') and table['utc']<checkpoints['utc'] and table['protocol_sha256']==sha(S/'protocol.json') and prep['prediction_parity']['literal_enumeration_inherited_from']=='H3bc' and prep['prediction_parity']['original_audit']==census['audit'] and prep['prediction_parity']['original_summary_sha256']==sha(B/'summary.json'))
    sourcepred=np.array(sourcepred,bool);predictions=np.unpackbits(targetbits,axis=1,count=73712,bitorder='little').astype(bool);lookup={bytes(row):i for i,row in enumerate(ordinary)};cache={}
    def choice(candidates,queried):
        key=(tuple(candidates),tuple(queried))
        if key in cache:return cache[key]
        best=min(int(scores[cid]) for cid in candidates);mapties=[cid for cid in candidates if int(scores[cid])==best];guide=min(mapties);count=np.count_nonzero(predictions[candidates],axis=0);dis=np.where(predictions[guide],len(candidates)-count,count);dis[queried]=-1;maximum=int(dis.max());ties=np.flatnonzero(dis==maximum);index=int(ties[0]) if maximum>0 else None;selection=dict(MAP_tied_candidates=mapties,guide_candidate=guide,MAP_integer_score=str(best),maximum_disagreements=maximum,optimal_board_count=len(ties),query_index=index,candidate_predictions={str(cid):bool(predictions[cid,index]) for cid in candidates} if index is not None else None,stop_reason=None if index is not None else 'no_informative_unqueried_board_remaining_identity_unresolved');cache[key]=selection;return selection
    expected_checkpoints=[];original_replayed=0
    for case in oldcases:
        truth=case['candidate'];source=parent_events[case['source_sequence']];indices=np.random.default_rng(case['seed']).choice(len(sourcepop[truth]),16,replace=False);sample=sourcepop[truth][indices];initial=np.flatnonzero(sourcepred[:,[lookup[bytes(row)] for row in sample]].all(1)).tolist();check(f'oldcase{case["case"]}source',source['kind']=='source_draw' and source['sha256']==case['source_record_sha256'] and source['data']['indices']==indices.tolist() and source['data']['columns_zero_based']==sample.tolist() and source['data']['initial_candidates']==initial==case['initial_candidates']);candidates=list(initial);queried=[]
        for oldstep in case['steps']:
            q=parent_events[oldstep['query_sequence']];a=parent_events[oldstep['answer_sequence']];selection=choice(candidates,queried);index=selection['query_index'];label=bool(predictions[truth,index]);remaining=[cid for cid in candidates if bool(predictions[cid,index])==label];request=dict(bits_path='predictions-bits.npy',scores_path='score-values.json',board_count=73712,candidates=candidates,queried=queried)
            check(f'oldcase{case["case"]}step{oldstep["step"]}exactoriginalpolicyandanswers',q['kind']=='query' and a['kind']=='answer' and q['sha256']==oldstep['query_sha256'] and a['sha256']==oldstep['answer_sha256'] and a['sequence']==q['sequence']+1 and q['data']['request']==request and q['data']['selection']==selection and oldstep['query_index']==index and oldstep['label']==a['data']['label']==label and oldstep['candidates_before']==a['data']['candidates_before']==candidates and oldstep['candidates_after']==a['data']['candidates_after']==remaining);candidates=remaining;queried=queried+[index];original_replayed+=1
        case_event=case_events[case['case']];check(f'oldcase{case["case"]}finalcheckpoint',case_event['data']==case and candidates==case['remaining_candidates'] and case['exact_true_singleton']==(candidates==[truth]) and (case['exact_true_singleton'] or len(queried)==4));expected_checkpoints.append(dict(case=case['case'],candidate=truth,bank=case['bank'],seed=case['seed'],original_case_record_sequence=case_event['sequence'],original_case_record_sha256=case_event['sha256'],original_source_record_sha256=source['sha256'],inherited_membership_labels=len(queried),original_four_query_exact=case['exact_true_singleton'],remaining_candidates=candidates,queried_board_indices=queried,original_query_hashes=[x['query_sha256'] for x in case['steps']],original_answer_hashes=[x['answer_sha256'] for x in case['steps']]))
    check('all5372originaloutputsand1408checkpointsbeforecontinuation',original_replayed==5372 and checkpoints['checkpoints']==expected_checkpoints and checkpoints['all5372_original_selector_outputs_identical'] and checkpoints['all1408_source_seeds_filters_and_final_states_reconstructed'] and checkpoints['original_membership_labels']==5372)
    at=0
    def take(kind):
        nonlocal at
        event=events[at];at+=1;check(f'new_event{at-1}kind',event['kind']==kind);return event
    first=take('study_start');check('newledgerparentbindingandpreamble',first['data']==dict(protocol_sha256=sha(S/'protocol.json'),preparation_sha256=sha(P/'preparation.json'),parent_ledger_raw_sha256=checkpoints['parent_ledger_raw_sha256'],parent_summary_sha256=checkpoints['parent_summary_sha256'],checkpoints_sha256=sha(P/'continuation-checkpoints.json')) and prep['utc']<first['utc']);computed=[];new_queries=0;stops=0
    for checkpoint in expected_checkpoints:
        case=checkpoint['case'];truth=checkpoint['candidate'];record=take('continuation_checkpoint');check(f'case{case}immutableparentcheckpoint',record['data']==checkpoint);candidates=list(checkpoint['remaining_candidates']);queried=list(checkpoint['queried_board_indices']);inherited=checkpoint['inherited_membership_labels'];steps=[];terminal=None
        for step in range(inherited,8):
            if len(candidates)==1:break
            selection=choice(candidates,queried);index=selection['query_index'];q=take('query' if index is not None else 'no_informative_query');request=dict(bits_path='predictions-bits.npy',scores_path='score-values.json',board_count=73712,candidates=candidates,queried=queried);check(f'newcase{case}step{step}exactunchangedpolicyandnohiddeninputs',q['data']==dict(case=case,step=step,request=request,selection=selection,board_zero_based=boards[index].tolist() if index is not None else None))
            if index is None:terminal=dict(step=step,decision=selection,record_sequence=q['sequence'],record_sha256=q['sha256']);stops+=1;break
            a=take('answer');label=bool(predictions[truth,index]);remaining=[cid for cid in candidates if bool(predictions[cid,index])==label];check(f'newcase{case}step{step}sealedanswerallcandidatesnorepeats',a['data']==dict(case=case,step=step,query_sha256=q['sha256'],label=label,candidates_before=candidates,candidates_after=remaining) and a['sequence']==q['sequence']+1 and q['monotonic_ns']<a['monotonic_ns'] and index not in queried and truth in remaining);steps.append(dict(step=step,query_index=index,label=label,candidates_before=candidates,candidates_after=remaining,query_sequence=q['sequence'],query_sha256=q['sha256'],answer_sequence=a['sequence'],answer_sha256=a['sha256']));candidates=remaining;queried=queried+[index];new_queries+=1
        result=dict(case=case,candidate=truth,bank=checkpoint['bank'],seed=checkpoint['seed'],original_four_query_exact=checkpoint['original_four_query_exact'],original_case_record_sha256=checkpoint['original_case_record_sha256'],original_remaining_candidates=checkpoint['remaining_candidates'],inherited_membership_labels=inherited,additional_membership_labels=len(steps),cumulative_membership_labels=inherited+len(steps),source_positive_labels=16,cumulative_observations=16+inherited+len(steps),remaining_candidates=candidates,exact_true_singleton=candidates==[truth],checkpoint_sequence=record['sequence'],checkpoint_record_sha256=record['sha256'],terminal_no_information=terminal,additional_steps=steps);finish=take('case_result');check(f'case{case}retainedresultstoppingandbudget',finish['data']==result==cases[case] and len(steps)<=4 and inherited+len(steps)<=8 and (not checkpoint['original_four_query_exact'] or (not steps and candidates==checkpoint['remaining_candidates'])));computed.append(result)
    truth_results=[]
    for cfg in p['truths']:
        rows=[r for r in computed if r['candidate']==cfg['candidate']];count=sum(r['exact_true_singleton'] for r in rows);truth_results.append(dict(candidate=cfg['candidate'],orbits=cfg['orbits'],cases=64,original_four_query_exact=sum(r['original_four_query_exact'] for r in rows),eight_query_exact=count,primary_required=48,primary_decision='supported' if count>=48 else 'falsified',secondary_required=61,secondary_decision='supported' if count>=61 else 'falsified',inherited_membership_labels=sum(r['inherited_membership_labels'] for r in rows),additional_membership_labels=sum(r['additional_membership_labels'] for r in rows),cumulative_membership_labels=sum(r['cumulative_membership_labels'] for r in rows)))
    decision='supported' if all(r['primary_decision']=='supported' for r in truth_results) else 'falsified';secondary='supported' if all(r['secondary_decision']=='supported' for r in truth_results) else 'falsified';total=sum(r['exact_true_singleton'] for r in computed);finish=take('study_complete');check('all1408casesseparateprimariesandlabelaccounting',at==len(events) and finish['data']==dict(decision=decision,secondary_decision=secondary,truth_results=truth_results,identified_cases=total,inherited_membership_labels=5372,additional_membership_labels=new_queries,cumulative_membership_labels=5372+new_queries) and g['truth_results']==truth_results and g['decision']==decision and g['secondary_decision']==secondary and g['cases']==1408 and g['original_identified_cases']==704 and g['identified_cases']==total and g['inherited_membership_labels']==5372 and g['additional_membership_labels']==new_queries and g['cumulative_membership_labels']==5372+new_queries)
    if require_summary:
        s=read(P/'summary.json');keys=['decision','secondary_decision','truth_results','cases','original_identified_cases','identified_cases','inherited_membership_labels','additional_membership_labels','cumulative_membership_labels','new_source_draws','ledger_records','ledger_sha256'];check('summaryidentities',s['status']=='validated' and all(s[k]==g[k] for k in keys) and s['protocol_sha256']==sha(S/'protocol.json') and s['audit_source_sha256']==sha(S/'analyze.py') and s['verifier_source_sha256']==sha(S/'verify_public.py'))
        for rel,digest in s['public_sha256'].items():check('export '+rel,sha(P/rel)==digest)
        plot=read(P/'plot-provenance.json');check('figureprovenance',all(sha(R/rel)==digest for rel,digest in plot['input_sha256'].items()) and all(sha(P/rel)==digest for rel,digest in plot['output_sha256'].items()))
    return dict(status='passed',checks=len(checks),full_literal_census_and_prior_durability_gate_inherited_by_hash=True,new_literal_enumerations=0,all1541_fresh_graph_filters_scores_verified=True,all5372_original_decisions_and1408_checkpoints_independently_replayed=True,all_new_query_before_answer_events_replayed=new_queries,both_ledger_hash_order_timestamp_chains_verified=True,all704_prior_successes_unchanged_without_newlabels=True,separate48_primary_and61_secondary_and_labels_verified=True,no_information_stops_verified=stops,elapsed_seconds=time.perf_counter()-start)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4]);print(json.dumps(verify(parser.parse_args().root)))


