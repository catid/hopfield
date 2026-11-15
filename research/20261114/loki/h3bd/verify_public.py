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


def verify(root,require_summary=True):
    start=time.perf_counter();R=Path(root);S=R/'research/20261114/loki/h3bd';P=R/'results/research/loki/h3bd';B=R/'results/research/loki/h3bc';read=lambda path:json.loads(path.read_text());sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest();checks=[]
    def check(name,value):assert value,name;checks.append(name)
    p=read(S/'protocol.json');g=read(P/'capacity-gate.json');raw=read(P/'study-result.json');prep=read(P/'preparation.json');old=read(R/p['normalizers']['selection_path']);previous=read(B/'summary.json');oldprep=read(B/'preparation.json');oldprotocol=read(R/'research/20261114/loki/h3bc/protocol.json')
    check('wholebudgetandcompletion',g['status']==raw['status']=='complete_verified' and not g['monitor_expired'] and g['aggregate_cpu_seconds']<=120 and g['wall_seconds']<=180 and p['budget']['GPUs']==0)
    check('freezeandstagedhashes',p['frozen_utc']<prep['utc']<raw['utc'] and sha(S/'protocol.json')==g['protocol_sha256']==prep['protocol_sha256'] and sha(S/'study.py')==g['source_sha256'] and sha(P/'preparation.json')==g['preparation_sha256'] and all(g[k]==raw[k] for k in raw if k!='utc'))
    for category in ['source_sha256','source_data_sha256']:
        for rel,digest in p[category].items():check(category+' '+rel,sha(R/rel)==digest)
    check('all108casesgridandoriginalcensuscoverage',p['cases']==oldprotocol['cases'] and len(p['cases'])==108 and p['inherited_truth_grid']==g['inherited_truth_grid']==oldprotocol['inherited_truth_grid'] and previous['status']=='validated' and previous['audit']['all1541N12_masks_and_N13_literal_populations_verified'] and previous['verifier_source_sha256']==sha(R/'research/20261114/loki/h3bc/verify_public.py') and previous['public_sha256']['preparation.json']==sha(B/'preparation.json') and previous['public_sha256']['predictions-bits.npy']==sha(B/'predictions-bits.npy') and previous['public_sha256']['ordinary13.npy']==sha(B/'ordinary13.npy'))
    orbits=list(itertools.combinations(range(1,12),2));choices=[x for k in [0,1,2] for x in itertools.combinations(range(55),k)];ordinary=np.load(R/p['normalizers']['ordinary_path']);boards=np.load(B/'ordinary13.npy');sourcebits=np.load(R/p['normalizers']['bits_path'])['bits'];targetbits=np.load(B/'predictions-bits.npy');hit12=hits(ordinary,orbits);hit13=hits(boards,orbits);base12,extra12=parts(12,orbits);base13,extra13=parts(13,orbits);scores=[];score_records=[];sourcepred=[];sourcepop={}
    check('inheritedordinaryuniversesandorder',ordinary.shape==(14200,12) and boards.shape==(73712,13) and all(bytes(a)<bytes(b) for a,b in zip(boards,boards[1:])))
    for cid,chosen in enumerate(choices):
        source_mask=~hit12[:,chosen].any(1) if chosen else np.ones(len(ordinary),bool);target_mask=~hit13[:,chosen].any(1) if chosen else np.ones(len(boards),bool);F12=base12.copy();F13=base13.copy()
        for j in chosen:F12|=extra12[j];F13|=extra13[j]
        columns=ordinary[source_mask];M=len(columns);sourcepop[cid]=columns;sourcepred.append(source_mask);record=dict(candidate=cid,orbit_indices=list(chosen),orbits=[list(orbits[j]) for j in chosen],complete_source_cardinality=M,source_columns_bytes_sha256=hashlib.sha256(columns.tobytes()).hexdigest(),source_F_bytes_sha256=hashlib.sha256(F12.tobytes()).hexdigest())
        check(f'candidate{cid}sourcefullpopulationandmetadata',np.array_equal(np.packbits(source_mask,bitorder='little'),sourcebits[cid]) and record==old['candidates'][cid])
        packed=np.packbits(target_mask,bitorder='little');target_record=dict(candidate=cid,states=int(target_mask.sum()),columns_bytes_sha256=hashlib.sha256(boards[target_mask].tobytes()).hexdigest(),F_bytes_sha256=hashlib.sha256(F13.tobytes()).hexdigest(),packed_bytes_sha256=hashlib.sha256(packed.tobytes()).hexdigest())
        check(f'candidate{cid}freshgraphfilterparity',np.array_equal(packed,targetbits[cid]) and target_record==oldprep['populations'][cid])
        value=math.comb(55,len(chosen))*math.comb(M,16) if M>=16 else None;scores.append(str(value) if value is not None else None);score_records.append(dict(candidate=cid,orbit_count=len(chosen),source_population=M,integer_score_without_common3=str(value) if value is not None else None))
    table=read(P/'score-table.json');check('exactpriorandallsourceinteger_scores',read(P/'score-values.json')==scores and table['scores']==score_records and p['frozen_utc']<table['utc']<prep['utc'] and table['protocol_sha256']==sha(S/'protocol.json') and prep['scores_sha256']==sha(P/'score-values.json') and prep['score_table_sha256']==sha(P/'score-table.json'))
    parity=prep['prediction_parity'];check('originalcensusauditexplicitlyinherited',parity['literal_enumeration_inherited_from']=='H3bc' and parity['original_audit']==previous['audit'] and parity['original_summary_sha256']==sha(B/'summary.json') and parity['all1541_graph_filters_and_metadata_exact'])
    sourcepred=np.array(sourcepred,bool);predictions=np.unpackbits(targetbits,axis=1,count=73712,bitorder='little').astype(bool);lookup={bytes(row):i for i,row in enumerate(ordinary)};results=[];queries=0;stops=0;same_state={}
    for cfg in p['cases']:
        case=cfg['case'];truth=cfg['true_candidate'];folder=P/f'cases/case{case}';sample=np.load(B/f'cases/case{case}/source-columns.npy');draw=np.random.default_rng(cfg['seed']).choice(len(sourcepop[truth]),64 if cfg['archive']=='H3ao' else 16,replace=False)[:16];sampleids=[lookup[bytes(row)] for row in sample];initial=np.flatnonzero(sourcepred[:,sampleids].all(1)).tolist()
        check(f'case{case}source_seed_filter_and_paired_bank',np.array_equal(sample,sourcepop[truth][draw]) and initial==previous['results'][case]['initial_candidates'] and previous['results'][case]['true_candidate']==truth and len(set(map(bytes,sample)))==16 and truth in initial and all(scores[cid] is not None for cid in initial))
        candidates=list(initial);steps=[];queried=[];terminal=None
        for step in range(4):
            if len(candidates)==1:break
            best=min(int(scores[cid]) for cid in candidates);mapties=[cid for cid in candidates if int(scores[cid])==best];guide=min(mapties);positive_count=np.count_nonzero(predictions[candidates],axis=0);disagreements=np.where(predictions[guide],len(candidates)-positive_count,positive_count);disagreements[queried]=-1;best_disagreement=int(disagreements.max());options=np.flatnonzero(disagreements==best_disagreement);index=int(options[0]) if best_disagreement>0 else None
            selection=dict(MAP_tied_candidates=mapties,guide_candidate=guide,MAP_integer_score=str(best),maximum_disagreements=best_disagreement,optimal_board_count=len(options),query_index=index,candidate_predictions={str(cid):bool(predictions[cid,index]) for cid in candidates} if index is not None else None,stop_reason=None if index is not None else 'no_informative_unqueried_board_remaining_identity_unresolved')
            qfile=folder/f'q{step}-sealed.json' if index is not None else folder/'no-informative-query.json';q=read(qfile);request=dict(bits_path='predictions-bits.npy',scores_path='score-values.json',board_count=73712,candidates=candidates,queried=queried)
            check(f'case{case}step{step}onlypermittedselectorinputs',q['request']==request)
            check(f'case{case}step{step}exactMAPtiesmaxdisagreementlexties',q['selection']==selection and q['step']==step and q['board_zero_based']==(boards[index].tolist() if index is not None else None) and q['protocol_sha256']==sha(S/'protocol.json') and prep['utc']<q['utc'])
            key=(tuple(candidates),tuple(queried));check(f'case{case}step{step}truthblinddecision',key not in same_state or same_state[key]==index);same_state[key]=index
            if index is None:terminal=dict(step=step,decision=selection,receipt_sha256=sha(qfile));stops+=1;break
            answerfile=folder/f'q{step}-answer.json';answer=read(answerfile);label=bool(predictions[truth,index]);remaining=[cid for cid in candidates if bool(predictions[cid,index])==label]
            check(f'case{case}step{step}sealbeforeanswer',answer['query_sha256']==sha(qfile) and q['utc']<=answer['utc'] and q['monotonic_ns']<answer['monotonic_ns'])
            check(f'case{case}step{step}oracleandallconsistentcandidates',answer['label']==label and answer['candidates_before']==candidates and answer['candidates_after']==remaining and truth in remaining and index not in queried)
            steps.append(dict(step=step,query_index=index,label=label,candidates_before=candidates,candidates_after=remaining,query_sha256=sha(qfile),answer_sha256=sha(answerfile)));candidates=remaining;queried=queried+[index];queries+=1
        check(f'case{case}exactstoppingandnoextraqueries',len(list(folder.glob('q*-sealed.json')))==len(steps) and len(list(folder.glob('q*-answer.json')))==len(steps) and (folder/'no-informative-query.json').exists()==(terminal is not None))
        exact=candidates==[truth];balanced=previous['results'][case]['exact_true_singleton'];pair='both_pass' if exact and balanced else 'guided_only' if exact else 'balanced_only' if balanced else 'both_fail';r=dict(case=case,archive=cfg['archive'],family=cfg['family'],bank=cfg['bank'],seed=cfg['seed'],true_candidate=truth,initial_candidates=initial,remaining_candidates=candidates,exact_true_singleton=exact,membership_labels_used=len(steps),positive_membership_labels=sum(x['label'] for x in steps),negative_membership_labels=sum(not x['label'] for x in steps),source_positive_labels=16,total_labels=16+len(steps),balanced_exact=balanced,paired_outcome=pair,terminal_no_information=terminal,steps=steps)
        check(f'case{case}retained_result',r==read(folder/'result.json'));results.append(r)
    total=sum(r['exact_true_singleton'] for r in results);single=sum(r['exact_true_singleton'] for r in results if r['archive']=='H3ao');endpoints=dict(total_at_least81='supported' if total>=81 else 'falsified',all8_singletons='supported' if single==8 else 'falsified');decision='supported' if all(v=='supported' for v in endpoints.values()) else 'falsified';paircounts={key:sum(r['paired_outcome']==key for r in results) for key in ['both_pass','guided_only','balanced_only','both_fail']}
    check('all108pairedoutcomesandseparateprimaries',results==g['results'] and total==g['identified'] and single==g['singleton_identified'] and queries==g['membership_labels'] and endpoints==g['primary_endpoints'] and decision==g['decision'] and g['secondary_all108']==('supported' if total==108 else 'falsified') and paircounts==g['paired_counts'])
    if require_summary:
        s=read(P/'summary.json');keys=['decision','primary_endpoints','secondary_all108','results','inherited_truth_grid','identified','singleton_identified','membership_labels','paired_counts'];check('summaryidentities',s['status']=='validated' and all(s[k]==g[k] for k in keys) and s['protocol_sha256']==sha(S/'protocol.json') and s['audit_source_sha256']==sha(S/'analyze.py') and s['verifier_source_sha256']==sha(S/'verify_public.py'))
        for rel,digest in s['public_sha256'].items():check('export '+rel,sha(P/rel)==digest)
        plot=read(P/'plot-provenance.json');check('figureprovenance',all(sha(R/rel)==digest for rel,digest in plot['input_sha256'].items()) and all(sha(P/rel)==digest for rel,digest in plot['output_sha256'].items()))
    return dict(status='passed',checks=len(checks),full_literal_census_inherited_from_hashpinned_H3bc=True,new_literal_enumerations=0,all1541_fresh_independent_graph_filters_and_exact_source_scores_verified=True,all108_raw_source_seeds_and_paired_outcomes_verified=True,all_sealed_queries_replayed=queries,no_information_stops_verified=stops,elapsed_seconds=time.perf_counter()-start)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4]);print(json.dumps(verify(parser.parse_args().root)))
