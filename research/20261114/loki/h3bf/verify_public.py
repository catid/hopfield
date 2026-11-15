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
    start=time.perf_counter();R=Path(root);S=R/'research/20261114/loki/h3bf';P=R/'results/research/loki/h3bf';B=R/'results/research/loki/h3bc';read=lambda path:json.loads(path.read_text());sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest();checks=[]
    def check(name,value):assert value,name;checks.append(name)
    p=read(S/'protocol.json');g=read(P/'capacity-gate.json');raw=read(P/'study-result.json');prep=read(P/'preparation.json');old=read(R/p['normalizers']['selection_path']);census=read(B/'summary.json');oldprep=read(B/'preparation.json');tail=read(R/'results/research/loki/h3be/summary.json');gate=read(P/'ledger-gate.json');cases=read(P/'case-results.json');export=read(P/'ledger-export.json');ledger_bytes=gzip.decompress((P/'query-ledger.jsonl.gz').read_bytes());events=chain(ledger_bytes)
    check('wholebudgetandcompletion',g['status']==raw['status']=='complete_verified' and not g['monitor_expired'] and g['aggregate_cpu_seconds']<=600 and g['wall_seconds']<=900 and p['budget']['GPUs']==0)
    check('stageidentitiesandrawledgerexport',p['frozen_utc']<gate['utc']<prep['utc']<raw['utc'] and sha(S/'protocol.json')==g['protocol_sha256']==prep['protocol_sha256'] and sha(S/'study.py')==g['source_sha256'] and sha(P/'preparation.json')==g['preparation_sha256'] and sha(P/'ledger-gate.json')==prep['ledger_gate_sha256'] and sha(P/'case-results.json')==g['case_results_sha256'] and all(g[k]==raw[k] for k in raw if k!='utc') and hashlib.sha256(ledger_bytes).hexdigest()==g['ledger_sha256']==export['raw_sha256'] and export['gzip_sha256']==sha(P/'query-ledger.jsonl.gz') and export['raw_bytes']==len(ledger_bytes) and len(events)==g['ledger_records'] and events[-1]['sha256']==g['ledger_last_record_sha256'])
    for category in ['source_sha256','source_data_sha256']:
        for rel,digest in p[category].items():check(category+' '+rel,sha(R/rel)==digest)
    selected=[r['candidate'] for r in tail['results'] if r['eligible'] and not r['exact_true_singleton']];prior=read(R/'research/20261114/loki/h3be/protocol.json');truths=[dict(candidate=cid,orbits=prior['truths'][cid]['orbits'],source_population=prior['truths'][cid]['source_population'],seeds=[2026200000+64*cid+bank for bank in range(64)]) for cid in selected]
    check('completepostselected22truths64frozenseeds',len(selected)==22 and truths==p['truths'] and sha(S/'selector.py')==p['unchanged_selector_sha256']==sha(R/'research/20261114/loki/h3be/selector.py'))
    check('inheritedfullcensus',census['status']=='validated' and census['audit']['all1541N12_masks_and_N13_literal_populations_verified'] and census['public_sha256']['predictions-bits.npy']==sha(B/'predictions-bits.npy') and census['public_sha256']['preparation.json']==sha(B/'preparation.json') and census['public_sha256']['ordinary13.npy']==sha(B/'ordinary13.npy'))
    orbits=list(itertools.combinations(range(1,12),2));choices=[x for k in [0,1,2] for x in itertools.combinations(range(55),k)];ordinary=np.load(R/p['normalizers']['ordinary_path']);boards=np.load(B/'ordinary13.npy');sourcebits=np.load(R/p['normalizers']['bits_path'])['bits'];targetbits=np.load(B/'predictions-bits.npy');hit12=hits(ordinary,orbits);hit13=hits(boards,orbits);base12,extra12=parts(12,orbits);base13,extra13=parts(13,orbits);scores=[];score_records=[];sourcepred=[];sourcepop={}
    for cid,chosen in enumerate(choices):
        m12=~hit12[:,chosen].any(1) if chosen else np.ones(len(ordinary),bool);m13=~hit13[:,chosen].any(1) if chosen else np.ones(len(boards),bool);F12=base12.copy();F13=base13.copy()
        for j in chosen:F12|=extra12[j];F13|=extra13[j]
        columns=ordinary[m12];M=len(columns);sourcepop[cid]=columns;sourcepred.append(m12);record=dict(candidate=cid,orbit_indices=list(chosen),orbits=[list(orbits[j]) for j in chosen],complete_source_cardinality=M,source_columns_bytes_sha256=hashlib.sha256(columns.tobytes()).hexdigest(),source_F_bytes_sha256=hashlib.sha256(F12.tobytes()).hexdigest())
        check(f'candidate{cid}sourcefullmaskandmetadata',np.array_equal(np.packbits(m12,bitorder='little'),sourcebits[cid]) and record==old['candidates'][cid])
        packed=np.packbits(m13,bitorder='little');r=dict(candidate=cid,states=int(m13.sum()),columns_bytes_sha256=hashlib.sha256(boards[m13].tobytes()).hexdigest(),F_bytes_sha256=hashlib.sha256(F13.tobytes()).hexdigest(),packed_bytes_sha256=hashlib.sha256(packed.tobytes()).hexdigest());check(f'candidate{cid}freshgraphfilter',np.array_equal(packed,targetbits[cid]) and r==oldprep['populations'][cid])
        score=math.comb(55,len(chosen))*math.comb(M,16) if M>=16 else None;scores.append(str(score) if score is not None else None);score_records.append(dict(candidate=cid,orbit_count=len(chosen),source_population=M,integer_score_without_common3=str(score) if score is not None else None))
    table=read(P/'score-table.json');parity=prep['prediction_parity'];check('allscoresandexplicitinheritedcoverage',table['scores']==score_records and read(P/'score-values.json')==scores and prep['scores_sha256']==sha(P/'score-values.json') and prep['score_table_sha256']==sha(P/'score-table.json') and table['utc']<gate['utc'] and table['protocol_sha256']==sha(S/'protocol.json') and parity['literal_enumeration_inherited_from']=='H3bc' and parity['original_audit']==census['audit'] and parity['original_summary_sha256']==sha(B/'summary.json'))
    synthetic=chain((P/'synthetic-ledger.jsonl').read_bytes());check('syntheticgateandformatcosts',len(synthetic)==16 and gate['synthetic_ledger_sha256']==sha(P/'synthetic-ledger.jsonl') and gate['source_sha256']==sha(S/'ledger_gate.py') and gate['ledger_source_sha256']==sha(S/'ledger.py') and gate['all8_prior_format_logical_queries_and_answers_identical'] and gate['oracle_callbacks_observed_durable_query_before_label']==8 and gate['tamper_cases_rejected']=={k:True for k in ['changed_answer_without_rehash','reordered_query_answer','broken_hash_chain','rehashed_wrong_query_reference']})
    for i,(qpath,apath) in enumerate(p['ledger_gate']['archival_query_answer_paths']):
        q=read(R/qpath);a=read(R/apath);qr=synthetic[2*i];ar=synthetic[2*i+1];receipt=gate['paired_format_parity'][i];expectedq=dict(index=i,request=q['request'],selection=q['selection'],board_zero_based=q['board_zero_based']);expecteda=dict(index=i,label=a['label'],candidates_before=a['candidates_before'],candidates_after=a['candidates_after'])
        check(f'syntheticpair{i}legacyparitydurabilityordering',qr['kind']=='query' and ar['kind']=='answer' and qr['data']==read(P/f'legacy-format-gate/q{i}.json')==expectedq and {k:v for k,v in ar['data'].items() if k!='query_sha256'}==read(P/f'legacy-format-gate/a{i}.json')==expecteda and receipt==dict(archival_query_sha256=sha(R/qpath),archival_answer_sha256=sha(R/apath),new_query_record_sha256=qr['sha256'],new_answer_record_sha256=ar['sha256'],logical_fields_identical=True,oracle_callback_monotonic_ns=receipt['oracle_callback_monotonic_ns']) and qr['monotonic_ns']<receipt['oracle_callback_monotonic_ns']<ar['monotonic_ns'])
    # Independent parser must reject altered, reordered and rehashed-invalid references.
    encode=lambda rows:b''.join(json.dumps(row,sort_keys=True,separators=(',',':')).encode()+b'\n' for row in rows);clone=lambda:json.loads(json.dumps(synthetic));mutants=[]
    x=clone();x[1]['data']['label']=not x[1]['data']['label'];mutants.append(x);x=clone();x[0],x[1]=x[1],x[0];mutants.append(x);x=clone();x[1]['previous_sha256']='1'*64;mutants.append(x);x=clone();x[1]['data']['query_sha256']='2'*64
    for i,row in enumerate(x):row.pop('sha256');row['previous_sha256']=x[i-1]['sha256'] if i else '0'*64;row['sha256']=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    mutants.append(x)
    for i,mutant in enumerate(mutants):
        try:chain(encode(mutant))
        except AssertionError:check(f'independenttampercase{i}',True)
        else:raise AssertionError('mutant accepted')
    sourcepred=np.array(sourcepred,bool);predictions=np.unpackbits(targetbits,axis=1,count=73712,bitorder='little').astype(bool);lookup={bytes(row):i for i,row in enumerate(ordinary)};at=0
    def take(kind):
        nonlocal at
        event=events[at];at+=1;check(f'event{at-1}kind',event['kind']==kind);return event
    first=take('study_start');check('studyledgerstartsaftergate',first['data']==dict(protocol_sha256=sha(S/'protocol.json'),preparation_sha256=sha(P/'preparation.json')) and prep['utc']<first['utc']);computed=[];truth_results=[];queries=0;stops=0;same_state={}
    for cfg in truths:
        truth=cfg['candidate'];truth_cases=[];keys=[]
        for bank,seed in enumerate(cfg['seeds']):
            case=len(computed);indices=np.random.default_rng(seed).choice(len(sourcepop[truth]),16,replace=False);sample=sourcepop[truth][indices];sampleids=[lookup[bytes(row)] for row in sample];initial=np.flatnonzero(sourcepred[:,sampleids].all(1)).tolist();key=hashlib.sha256(b''.join(sorted(map(bytes,sample)))).hexdigest();keys.append(key);source=take('source_draw');expectedsource=dict(case=case,candidate=truth,bank=bank,seed=seed,source_population=len(sourcepop[truth]),indices=indices.tolist(),columns_zero_based=sample.tolist(),initial_candidates=initial,unordered_sample_sha256=key)
            check(f'case{case}freshseedrawdrawandallcandidatefilter',source['data']==expectedsource and truth in initial and len(set(map(bytes,sample)))==16);candidates=list(initial);steps=[];queried=[];terminal=None
            for step in range(4):
                if len(candidates)==1:break
                best=min(int(scores[cid]) for cid in candidates);mapties=[cid for cid in candidates if int(scores[cid])==best];guide=min(mapties);count=np.count_nonzero(predictions[candidates],axis=0);dis=np.where(predictions[guide],len(candidates)-count,count);dis[queried]=-1;maximum=int(dis.max());ties=np.flatnonzero(dis==maximum);index=int(ties[0]) if maximum>0 else None;selection=dict(MAP_tied_candidates=mapties,guide_candidate=guide,MAP_integer_score=str(best),maximum_disagreements=maximum,optimal_board_count=len(ties),query_index=index,candidate_predictions={str(cid):bool(predictions[cid,index]) for cid in candidates} if index is not None else None,stop_reason=None if index is not None else 'no_informative_unqueried_board_remaining_identity_unresolved');request=dict(bits_path='predictions-bits.npy',scores_path='score-values.json',board_count=73712,candidates=candidates,queried=queried)
                query=take('query' if index is not None else 'no_informative_query');expectedq=dict(case=case,step=step,request=request,selection=selection,board_zero_based=boards[index].tolist() if index is not None else None);check(f'case{case}step{step}truthblindexactpolicyties',query['data']==expectedq);state=(tuple(candidates),tuple(queried));check(f'case{case}step{step}statelesscaseparity',state not in same_state or same_state[state]==index);same_state[state]=index
                if index is None:terminal=dict(step=step,decision=selection,record_sequence=query['sequence'],record_sha256=query['sha256']);stops+=1;break
                answer=take('answer');label=bool(predictions[truth,index]);remaining=[cid for cid in candidates if bool(predictions[cid,index])==label];expecteda=dict(case=case,step=step,query_sha256=query['sha256'],label=label,candidates_before=candidates,candidates_after=remaining);check(f'case{case}step{step}orderedsealedanswerandallcandidates',answer['data']==expecteda and answer['sequence']==query['sequence']+1 and query['monotonic_ns']<answer['monotonic_ns'] and truth in remaining and index not in queried)
                steps.append(dict(step=step,query_index=index,label=label,candidates_before=candidates,candidates_after=remaining,query_sequence=query['sequence'],query_sha256=query['sha256'],answer_sequence=answer['sequence'],answer_sha256=answer['sha256']));candidates=remaining;queried=queried+[index];queries+=1
            result=dict(case=case,candidate=truth,bank=bank,seed=seed,source_population=len(sourcepop[truth]),initial_candidates=initial,remaining_candidates=candidates,exact_true_singleton=candidates==[truth],membership_labels_used=len(steps),positive_membership_labels=sum(x['label'] for x in steps),negative_membership_labels=sum(not x['label'] for x in steps),source_positive_labels=16,total_labels=16+len(steps),unordered_sample_sha256=key,source_sequence=source['sequence'],source_record_sha256=source['sha256'],terminal_no_information=terminal,steps=steps);finish=take('case_result');check(f'case{case}alloutcomesandstop',finish['data']==result==cases[case]);computed.append(result);truth_cases.append(result)
        count=sum(r['exact_true_singleton'] for r in truth_cases);truth_results.append(dict(candidate=truth,orbits=cfg['orbits'],source_population=cfg['source_population'],banks=64,unique_unordered_source_banks=len(set(keys)),identified=count,required_exact=48,decision='supported' if count>=48 else 'falsified',membership_labels=sum(r['membership_labels_used'] for r in truth_cases)))
    total=sum(r['exact_true_singleton'] for r in computed);decision='supported' if all(r['decision']=='supported' for r in truth_results) else 'falsified';finish=take('study_complete');check('all1408cases22primarycountsandunique_banks',at==len(events) and len(cases)==1408 and finish['data']==dict(decision=decision,truth_results=truth_results,identified_cases=total,membership_labels=queries) and truth_results==g['truth_results'] and g['decision']==decision and g['cases']==1408 and g['identified_cases']==total and g['membership_labels']==queries)
    if require_summary:
        s=read(P/'summary.json');keys=['decision','truth_results','cases','identified_cases','membership_labels','ledger_records','ledger_sha256'];check('summaryidentities',s['status']=='validated' and all(s[k]==g[k] for k in keys) and s['protocol_sha256']==sha(S/'protocol.json') and s['audit_source_sha256']==sha(S/'analyze.py') and s['verifier_source_sha256']==sha(S/'verify_public.py'))
        for rel,digest in s['public_sha256'].items():check('export '+rel,sha(P/rel)==digest)
        plot=read(P/'plot-provenance.json');check('figureprovenance',all(sha(R/rel)==digest for rel,digest in plot['input_sha256'].items()) and all(sha(P/rel)==digest for rel,digest in plot['output_sha256'].items()))
    return dict(status='passed',checks=len(checks),full_literal_census_inherited_from_hashpinned_H3bc=True,new_literal_enumerations=0,all1541_fresh_graph_filters_and_scores_verified=True,all1408_source_seeds_and_unique_bank_counts_verified=True,entire_ledger_hash_order_timestamp_chain_verified=len(events),all_query_before_answer_and_policy_decisions_replayed=queries,synthetic_prior_format_parity_and_tamper_rejection_verified=True,no_information_stops_verified=stops,elapsed_seconds=time.perf_counter()-start)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4]);print(json.dumps(verify(parser.parse_args().root)))


