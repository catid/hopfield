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
import subprocess
import sys
import tempfile
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
    start=time.perf_counter();R=Path(root);S=R/'research/20261114/loki/h3be';P=R/'results/research/loki/h3be';B=R/'results/research/loki/h3bc';read=lambda path:json.loads(path.read_text());sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest();checks=[]
    def check(name,value):assert value,name;checks.append(name)
    p=read(S/'protocol.json');g=read(P/'capacity-gate.json');raw=read(P/'study-result.json');prep=read(P/'preparation.json');old=read(R/p['normalizers']['selection_path']);previous=read(B/'summary.json');oldprep=read(B/'preparation.json');runtime=read(P/'runtime-parity.json')
    check('wholebudgetandcompletion',g['status']==raw['status']=='complete_verified' and not g['monitor_expired'] and g['aggregate_cpu_seconds']<=600 and g['wall_seconds']<=900 and p['budget']['GPUs']==0)
    check('freezeandstagedhashes',p['frozen_utc']<runtime['utc']<prep['utc']<raw['utc'] and sha(S/'protocol.json')==g['protocol_sha256']==prep['protocol_sha256'] and sha(S/'study.py')==g['source_sha256'] and sha(P/'preparation.json')==g['preparation_sha256'] and sha(P/'runtime-parity.json')==prep['runtime_parity_sha256'] and all(g[k]==raw[k] for k in raw if k!='utc'))
    for category in ['source_sha256','source_data_sha256']:
        for rel,digest in p[category].items():check(category+' '+rel,sha(R/rel)==digest)
    check('originalcensuscoverage',previous['status']=='validated' and previous['audit']['all1541N12_masks_and_N13_literal_populations_verified'] and previous['verifier_source_sha256']==sha(R/'research/20261114/loki/h3bc/verify_public.py') and previous['public_sha256']['preparation.json']==sha(B/'preparation.json') and previous['public_sha256']['predictions-bits.npy']==sha(B/'predictions-bits.npy') and previous['public_sha256']['ordinary13.npy']==sha(B/'ordinary13.npy'))
    expectedtruths=[dict(candidate=c['candidate'],orbit_count=len(c['orbits']),orbits=c['orbits'],source_population=c['complete_source_cardinality'],eligible=c['complete_source_cardinality']>=16,seed=2026120000+c['candidate']) for c in old['candidates']];strata=[dict(orbit_count=k,eligible_truths=sum(t['eligible'] and t['orbit_count']==k for t in expectedtruths)) for k in [0,1,2]]
    for cfg in strata:cfg['required_exact']=(3*cfg['eligible_truths']+3)//4
    check('all1541truths1511eligible30ineligibleandfrozenseeds',p['truths']==expectedtruths and p['strata']==strata and len(expectedtruths)==1541 and sum(t['eligible'] for t in expectedtruths)==1511)
    orbits=list(itertools.combinations(range(1,12),2));choices=[x for k in [0,1,2] for x in itertools.combinations(range(55),k)];ordinary=np.load(R/p['normalizers']['ordinary_path']);boards=np.load(B/'ordinary13.npy');sourcebits=np.load(R/p['normalizers']['bits_path'])['bits'];targetbits=np.load(B/'predictions-bits.npy');hit12=hits(ordinary,orbits);hit13=hits(boards,orbits);base12,extra12=parts(12,orbits);base13,extra13=parts(13,orbits);scores=[];score_records=[];sourcepred=[];sourcepop={}
    check('inheritedordinaryuniversesandlexorder',ordinary.shape==(14200,12) and boards.shape==(73712,13) and all(bytes(a)<bytes(b) for a,b in zip(boards,boards[1:])))
    for cid,chosen in enumerate(choices):
        source_mask=~hit12[:,chosen].any(1) if chosen else np.ones(len(ordinary),bool);target_mask=~hit13[:,chosen].any(1) if chosen else np.ones(len(boards),bool);F12=base12.copy();F13=base13.copy()
        for j in chosen:F12|=extra12[j];F13|=extra13[j]
        columns=ordinary[source_mask];M=len(columns);sourcepop[cid]=columns;sourcepred.append(source_mask);record=dict(candidate=cid,orbit_indices=list(chosen),orbits=[list(orbits[j]) for j in chosen],complete_source_cardinality=M,source_columns_bytes_sha256=hashlib.sha256(columns.tobytes()).hexdigest(),source_F_bytes_sha256=hashlib.sha256(F12.tobytes()).hexdigest())
        check(f'candidate{cid}sourcefullpopulationandmetadata',np.array_equal(np.packbits(source_mask,bitorder='little'),sourcebits[cid]) and record==old['candidates'][cid])
        packed=np.packbits(target_mask,bitorder='little');target_record=dict(candidate=cid,states=int(target_mask.sum()),columns_bytes_sha256=hashlib.sha256(boards[target_mask].tobytes()).hexdigest(),F_bytes_sha256=hashlib.sha256(F13.tobytes()).hexdigest(),packed_bytes_sha256=hashlib.sha256(packed.tobytes()).hexdigest())
        check(f'candidate{cid}freshgraphfilterparity',np.array_equal(packed,targetbits[cid]) and target_record==oldprep['populations'][cid])
        value=math.comb(55,len(chosen))*math.comb(M,16) if M>=16 else None;scores.append(str(value) if value is not None else None);score_records.append(dict(candidate=cid,orbit_count=len(chosen),source_population=M,integer_score_without_common3=str(value) if value is not None else None))
    table=read(P/'score-table.json');check('exactpriorandallsourceinteger_scores',read(P/'score-values.json')==scores and table['scores']==score_records and p['frozen_utc']<table['utc']<runtime['utc'] and table['protocol_sha256']==sha(S/'protocol.json') and prep['scores_sha256']==sha(P/'score-values.json') and prep['score_table_sha256']==sha(P/'score-table.json'))
    parity=prep['prediction_parity'];check('originalcensusauditexplicitlyinherited',parity['literal_enumeration_inherited_from']=='H3bc' and parity['original_audit']==previous['audit'] and parity['original_summary_sha256']==sha(B/'summary.json') and parity['all1541_graph_filters_and_metadata_exact'])
    inherited_guided=read(R/'results/research/loki/h3bd/summary.json');archival=[f'results/research/loki/h3bd/cases/case{r["case"]}/q{x["step"]}-sealed.json' for r in inherited_guided['results'] for x in r['steps']]
    check('fixed278archivalruntimecases',len(archival)==278 and p['optimization']['archival_query_paths']==archival and runtime['all278_full_selection_dictionaries_identical'] and runtime['per_case_state_supplied_entirely_by_request'])
    with tempfile.TemporaryDirectory(prefix='h3be-selector-audit-') as temp:
        temp=Path(temp);np.save(temp/'predictions-bits.npy',targetbits);(temp/'score-values.json').write_text(json.dumps(scores));worker=subprocess.Popen([sys.executable,str(S/'selector.py')],cwd=temp,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        try:
            for i,rel in enumerate(archival):
                oldquery=read(R/rel);worker.stdin.write(json.dumps(oldquery['request'])+'\n');worker.stdin.flush();selection=json.loads(worker.stdout.readline());check(f'archivaldecision{i}persistentexactparity',selection==oldquery['selection'] and runtime['archival_decisions'][i]==dict(query_path=rel,query_sha256=sha(R/rel),full_selection_identical=True))
            worker.stdin.close();check('persistentselectorcleanexit',worker.wait()==0)
        finally:
            if worker.poll() is None:worker.terminate();worker.wait()
    sourcepred=np.array(sourcepred,bool);predictions=np.unpackbits(targetbits,axis=1,count=73712,bitorder='little').astype(bool);lookup={bytes(row):i for i,row in enumerate(ordinary)};results=[];queries=0;stops=0;same_state={}
    for cfg in expectedtruths:
        truth=cfg['candidate'];folder=P/f'cases/candidate{truth}'
        if not cfg['eligible']:
            r=dict(candidate=truth,orbit_count=cfg['orbit_count'],eligible=False,source_population=cfg['source_population'],seed=cfg['seed'],decision='ineligible_source_fewer_than16',exact_true_singleton=None,membership_labels_used=0,source_positive_labels=0,total_labels=0,remaining_candidates=None,steps=[]);check(f'candidate{truth}ineligiblewithoutdraworqueries',r==read(folder/'result.json') and sorted(x.name for x in folder.iterdir())==['result.json']);results.append(r);continue
        draw=np.load(folder/'draw.npz');indices=np.random.default_rng(cfg['seed']).choice(len(sourcepop[truth]),16,replace=False);sample=sourcepop[truth][indices];sampleids=[lookup[bytes(row)] for row in sample];initial=np.flatnonzero(sourcepred[:,sampleids].all(1)).tolist();evidence=read(folder/'source-evidence.json')
        check(f'candidate{truth}newseedandall1541positivefilter',np.array_equal(draw['indices'],indices) and np.array_equal(draw['columns'],sample) and len(set(map(bytes,sample)))==16 and evidence['initial_candidates']==initial and evidence['candidate']==truth and evidence['seed']==cfg['seed'] and evidence['source_population']==len(sourcepop[truth]) and evidence['source_positive_labels']==16 and evidence['draw_sha256']==sha(folder/'draw.npz') and evidence['protocol_sha256']==sha(S/'protocol.json') and prep['utc']<evidence['utc'] and truth in initial)
        candidates=list(initial);steps=[];queried=[];terminal=None
        for step in range(4):
            if len(candidates)==1:break
            best=min(int(scores[cid]) for cid in candidates);mapties=[cid for cid in candidates if int(scores[cid])==best];guide=min(mapties);positive_count=np.count_nonzero(predictions[candidates],axis=0);disagreements=np.where(predictions[guide],len(candidates)-positive_count,positive_count);disagreements[queried]=-1;best_disagreement=int(disagreements.max());options=np.flatnonzero(disagreements==best_disagreement);index=int(options[0]) if best_disagreement>0 else None
            selection=dict(MAP_tied_candidates=mapties,guide_candidate=guide,MAP_integer_score=str(best),maximum_disagreements=best_disagreement,optimal_board_count=len(options),query_index=index,candidate_predictions={str(cid):bool(predictions[cid,index]) for cid in candidates} if index is not None else None,stop_reason=None if index is not None else 'no_informative_unqueried_board_remaining_identity_unresolved')
            qfile=folder/f'q{step}-sealed.json' if index is not None else folder/'no-informative-query.json';q=read(qfile);request=dict(bits_path='predictions-bits.npy',scores_path='score-values.json',board_count=73712,candidates=candidates,queried=queried)
            check(f'candidate{truth}step{step}onlypermittedinputs',q['request']==request)
            check(f'candidate{truth}step{step}exactunchangedMAPandqueryties',q['selection']==selection and q['step']==step and q['board_zero_based']==(boards[index].tolist() if index is not None else None) and q['protocol_sha256']==sha(S/'protocol.json') and evidence['utc']<q['utc'])
            key=(tuple(candidates),tuple(queried));check(f'candidate{truth}step{step}caseindependenttruthblindpolicy',key not in same_state or same_state[key]==index);same_state[key]=index
            if index is None:terminal=dict(step=step,decision=selection,receipt_sha256=sha(qfile));stops+=1;break
            afile=folder/f'q{step}-answer.json';answer=read(afile);label=bool(predictions[truth,index]);remaining=[cid for cid in candidates if bool(predictions[cid,index])==label]
            check(f'candidate{truth}step{step}sealbeforeoracle',answer['query_sha256']==sha(qfile) and q['utc']<=answer['utc'] and q['monotonic_ns']<answer['monotonic_ns'])
            check(f'candidate{truth}step{step}labelandallcompatiblemodels',answer['label']==label and answer['candidates_before']==candidates and answer['candidates_after']==remaining and truth in remaining and index not in queried)
            steps.append(dict(step=step,query_index=index,label=label,candidates_before=candidates,candidates_after=remaining,query_sha256=sha(qfile),answer_sha256=sha(afile)));candidates=remaining;queried=queried+[index];queries+=1
        check(f'candidate{truth}exactstopping',len(list(folder.glob('q*-sealed.json')))==len(steps) and len(list(folder.glob('q*-answer.json')))==len(steps) and (folder/'no-informative-query.json').exists()==(terminal is not None))
        r=dict(candidate=truth,orbit_count=cfg['orbit_count'],eligible=True,source_population=cfg['source_population'],seed=cfg['seed'],decision='supported' if candidates==[truth] else 'falsified',initial_candidates=initial,remaining_candidates=candidates,exact_true_singleton=candidates==[truth],membership_labels_used=len(steps),positive_membership_labels=sum(x['label'] for x in steps),negative_membership_labels=sum(not x['label'] for x in steps),source_positive_labels=16,total_labels=16+len(steps),source_evidence_sha256=sha(folder/'source-evidence.json'),terminal_no_information=terminal,steps=steps)
        check(f'candidate{truth}retainedresult',r==read(folder/'result.json'));results.append(r)
    endpoints=[]
    for cfg in strata:
        rows=[r for r in results if r['eligible'] and r['orbit_count']==cfg['orbit_count']];count=sum(r['exact_true_singleton'] for r in rows);endpoints.append(dict(cfg,identified=count,decision='supported' if count>=cfg['required_exact'] else 'falsified',membership_labels=sum(r['membership_labels_used'] for r in rows)))
    check('all1541outcomesandseparatefinitestratumprimaries',g['results']==results and g['strata']==endpoints and g['decision']==('supported' if all(r['decision']=='supported' for r in endpoints) else 'falsified') and g['eligible_truths']==1511 and g['ineligible_truths']==30 and g['identified']==sum(r['exact_true_singleton'] for r in results if r['eligible']) and g['membership_labels']==queries)
    if require_summary:
        s=read(P/'summary.json');keys=['decision','strata','results','eligible_truths','ineligible_truths','identified','membership_labels'];check('summaryidentities',s['status']=='validated' and all(s[k]==g[k] for k in keys) and s['protocol_sha256']==sha(S/'protocol.json') and s['audit_source_sha256']==sha(S/'analyze.py') and s['verifier_source_sha256']==sha(S/'verify_public.py'))
        for rel,digest in s['public_sha256'].items():check('export '+rel,sha(P/rel)==digest)
        plot=read(P/'plot-provenance.json');check('figureprovenance',all(sha(R/rel)==digest for rel,digest in plot['input_sha256'].items()) and all(sha(P/rel)==digest for rel,digest in plot['output_sha256'].items()))
    return dict(status='passed',checks=len(checks),full_literal_census_inherited_from_hashpinned_H3bc=True,new_literal_enumerations=0,all1541_graph_filters_scores_trutheligibilities_verified=True,all1511_new_source_seeds_and_full_positive_filters_verified=True,all278_archival_decisions_replayed_in_persistent_selector=True,all_sealed_fresh_queries_replayed=queries,no_information_stops_verified=stops,all30_ineligibles_retained_without_draws=True,elapsed_seconds=time.perf_counter()-start)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4]);print(json.dumps(verify(parser.parse_args().root)))


