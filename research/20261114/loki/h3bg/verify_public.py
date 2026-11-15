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
    start=time.perf_counter();R=Path(root);S=R/'research/20261114/loki/h3bg';P=R/'results/research/loki/h3bg';B=R/'results/research/loki/h3bc';F=R/'results/research/loki/h3bf';read=lambda path:json.loads(path.read_text());sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest();checks=[]
    def check(name,value):assert value,name;checks.append(name)
    p=read(S/'protocol.json');g=read(P/'capacity-gate.json');raw=read(P/'study-result.json');partition=read(P/'N13-partition.json');actualpairs=read(P/'pair-difference-witnesses.json');actualcases=read(P/'case-ceilings.json');old=read(R/p['normalizers']['selection_path']);census=read(B/'summary.json');oldprep=read(B/'preparation.json');prior=read(F/'summary.json');oldcases=read(F/'case-results.json')
    check('wholebudgetandcompletion',g['status']==raw['status']=='complete_verified' and not g['monitor_expired'] and g['aggregate_cpu_seconds']<=120 and g['wall_seconds']<=180 and p['budget']['GPUs']==0 and g['new_oracle_labels']==0 and g['distinct_truth_competitor_pairs']<=33880)
    check('frozenstagehashes',p['frozen_utc']<partition['utc']<actualpairs['utc']<raw['utc'] and sha(S/'protocol.json')==g['protocol_sha256']==partition['protocol_sha256'] and sha(S/'study.py')==g['source_sha256'] and all(g[k]==raw[k] for k in raw if k!='utc') and sha(P/'N13-partition.json')==g['partition_sha256'] and sha(P/'case-ceilings.json')==g['cases_sha256'] and sha(P/'pair-difference-witnesses.json')==g['pair_witnesses_sha256'])
    for category in ['source_sha256','source_data_sha256']:
        for rel,digest in p[category].items():check(category+' '+rel,sha(R/rel)==digest)
    check('pinnedfullcensusandpriorledgeraudit',census['audit']['all1541N12_masks_and_N13_literal_populations_verified'] and census['public_sha256']['predictions-bits.npy']==sha(B/'predictions-bits.npy') and census['public_sha256']['ordinary13.npy']==sha(B/'ordinary13.npy') and census['public_sha256']['preparation.json']==sha(B/'preparation.json') and prior['audit']['all1408_source_seeds_and_unique_bank_counts_verified'] and prior['public_sha256']['case-results.json']==sha(F/'case-results.json') and prior['public_sha256']['query-ledger.jsonl.gz']==sha(F/'query-ledger.jsonl.gz') and prior['public_sha256']['ledger-export.json']==sha(F/'ledger-export.json'))
    check('all1408knownsuccessandfailurecasesfrozen',len(oldcases)==p['all_cases']==1408 and p['primary_failed_case_ids']==[r['case'] for r in oldcases if not r['exact_true_singleton']] and p['known_success_case_ids']==[r['case'] for r in oldcases if r['exact_true_singleton']] and len(p['primary_failed_case_ids'])==len(p['known_success_case_ids'])==704)
    orbits=list(itertools.combinations(range(1,12),2));choices=[x for k in [0,1,2] for x in itertools.combinations(range(55),k)];ordinary=np.load(R/p['normalizers']['ordinary_path']);boards=np.load(B/'ordinary13.npy');sourcebits=np.load(R/p['normalizers']['bits_path'])['bits'];archivebits=np.load(B/'predictions-bits.npy');h12=hits(ordinary,orbits);h13=hits(boards,orbits);base12,extra12=parts(12,orbits);base13,extra13=parts(13,orbits);sourcepred=[];sourcepop={};targetbits=[];population_records=[]
    check('ordinaryuniverseboundsandlexorder',ordinary.shape==(14200,12) and boards.shape==(73712,13) and all(bytes(a)<bytes(b) for a,b in zip(boards,boards[1:])))
    for cid,chosen in enumerate(choices):
        m12=~h12[:,chosen].any(1) if chosen else np.ones(len(ordinary),bool);m13=~h13[:,chosen].any(1) if chosen else np.ones(len(boards),bool);F12=base12.copy();F13=base13.copy()
        for j in chosen:F12|=extra12[j];F13|=extra13[j]
        columns=ordinary[m12];sourcepop[cid]=columns;sourcepred.append(m12);record=dict(candidate=cid,orbit_indices=list(chosen),orbits=[list(orbits[j]) for j in chosen],complete_source_cardinality=len(columns),source_columns_bytes_sha256=hashlib.sha256(columns.tobytes()).hexdigest(),source_F_bytes_sha256=hashlib.sha256(F12.tobytes()).hexdigest());check(f'candidate{cid}completeN12maskmetadata',np.array_equal(np.packbits(m12,bitorder='little'),sourcebits[cid]) and record==old['candidates'][cid])
        packed=np.packbits(m13,bitorder='little');targetbits.append(packed);record=dict(candidate=cid,states=int(m13.sum()),columns_bytes_sha256=hashlib.sha256(boards[m13].tobytes()).hexdigest(),F_bytes_sha256=hashlib.sha256(F13.tobytes()).hexdigest(),packed_bytes_sha256=hashlib.sha256(packed.tobytes()).hexdigest());check(f'candidate{cid}completeN13filtermetadata',np.array_equal(packed,archivebits[cid]) and record==oldprep['populations'][cid]);population_records.append(record)
    targetbits=np.array(targetbits,np.uint8);_,inverse=np.unique(targetbits,axis=0,return_inverse=True);members=sorted([np.flatnonzero(inverse==label).tolist() for label in np.unique(inverse)],key=lambda ids:ids[0]);classes=[];modelclass={}
    for i,ids in enumerate(members):
        record=dict(class_id=i,states=population_records[ids[0]]['states'],prediction_bits_sha256=population_records[ids[0]]['packed_bytes_sha256'],members=ids);classes.append(record)
        for cid in ids:modelclass[cid]=i
    check('independentfull1541predictionpartitionandempties',partition['classes']==classes and partition['candidate_to_class']==[modelclass[cid] for cid in range(1541)] and partition['all1541models_retained'] and partition['empty_models']==[r['candidate'] for r in population_records if r['states']==0])
    sourcepred=np.array(sourcepred,bool);lookup={bytes(row):i for i,row in enumerate(ordinary)};results=[];needed=set();truthconfig={t['candidate']:t for t in p['truths']}
    for case in oldcases:
        truth=case['candidate'];check(f'case{case["case"]}frozenpriorseed',truthconfig[truth]['seeds'][case['bank']]==case['seed']);indices=np.random.default_rng(case['seed']).choice(len(sourcepop[truth]),16,replace=False);sample=sourcepop[truth][indices];initial=np.flatnonzero(sourcepred[:,[lookup[bytes(row)] for row in sample]].all(1)).tolist();key=hashlib.sha256(b''.join(sorted(map(bytes,sample)))).hexdigest();aliases=[cid for cid in initial if modelclass[cid]==modelclass[truth]];identifiable=aliases==[truth];known=case['exact_true_singleton'];check(f'case{case["case"]}sourcefilterandknownsuccessconsistency',initial==case['initial_candidates'] and key==case['unordered_sample_sha256'] and (not known or identifiable));needed.update((truth,cid) for cid in initial if cid!=truth)
        results.append(dict(case=case['case'],candidate=truth,bank=case['bank'],seed=case['seed'],initial_candidates=initial,observed_four_query_exact=known,diagnostic_scope='known_success_consistency' if known else 'new_failed_case_ceiling',global_N13_class=modelclass[truth],N13_indistinguishable_initial_candidates=aliases,N13_identifiable_in_principle=identifiable,source_unordered_sample_sha256=key))
    check('all1408exactceilingintersections',results==actualcases);pairs=[];popcount=np.array([int(i).bit_count() for i in range(256)],np.uint8)
    for truth,other in sorted(needed):
        directions=[]
        for valid,invalid in [(truth,other),(other,truth)]:
            byte_difference=np.bitwise_and(targetbits[valid],np.bitwise_not(targetbits[invalid]));count=int(popcount[byte_difference].sum());nonzero=np.flatnonzero(byte_difference)
            if len(nonzero):byte=int(nonzero[0]);value=int(byte_difference[byte]);first=8*byte+(value&-value).bit_length()-1
            else:first=None
            directions.append(dict(valid_candidate=valid,invalid_candidate=invalid,states=count,lexfirst_board_index=first,lexfirst_columns_zero_based=boards[first].tolist() if first is not None else None))
        pair=dict(true_candidate=truth,other_initial_candidate=other,predictions_identical=modelclass[truth]==modelclass[other],directions=directions);check(f'pair{truth},{other}identityversusdirecteddifferences',pair['predictions_identical']==all(d['states']==0 for d in directions));pairs.append(pair)
    check('allrequiredpairsbothdirectionslexwitnessesandzeros',pairs==actualpairs['pairs'] and actualpairs['all_required_pairs_retained'] and actualpairs['truth_aware_diagnostic_not_query_policy'] and len(pairs)==g['distinct_truth_competitor_pairs'])
    truth_results=[]
    for cfg in p['truths']:
        rows=[r for r in results if r['candidate']==cfg['candidate']];failed=[r for r in rows if not r['observed_four_query_exact']];truth_results.append(dict(candidate=cfg['candidate'],orbits=cfg['orbits'],cases=len(rows),observed_four_query_exact=sum(r['observed_four_query_exact'] for r in rows),known_success_identifiable=sum(r['observed_four_query_exact'] and r['N13_identifiable_in_principle'] for r in rows),failed_cases=len(failed),failed_cases_identifiable=sum(r['N13_identifiable_in_principle'] for r in failed),all_case_identifiability_ceiling=sum(r['N13_identifiable_in_principle'] for r in rows),unavoidable_ambiguous_cases=sum(not r['N13_identifiable_in_principle'] for r in rows)))
    failed=[r for r in results if not r['observed_four_query_exact']];decision='supported' if all(r['N13_identifiable_in_principle'] for r in failed) else 'falsified';check('primaryandalltruthceilingsknownsuccessseparation',g['decision']==decision and g['truth_results']==truth_results and g['cases']==1408 and g['known_success_cases']==704 and g['failed_cases']==704 and g['failed_cases_identifiable']==sum(r['N13_identifiable_in_principle'] for r in failed) and g['unavoidable_ambiguous_cases']==sum(not r['N13_identifiable_in_principle'] for r in results) and g['global_N13_equivalence_classes']==len(classes) and g['global_nontrivial_classes']==sum(len(c['members'])>1 for c in classes))
    if require_summary:
        s=read(P/'summary.json');keys=['decision','truth_results','cases','known_success_cases','failed_cases','failed_cases_identifiable','unavoidable_ambiguous_cases','global_N13_equivalence_classes','global_nontrivial_classes','distinct_truth_competitor_pairs','new_oracle_labels'];check('summaryidentities',s['status']=='validated' and all(s[k]==g[k] for k in keys) and s['protocol_sha256']==sha(S/'protocol.json') and s['audit_source_sha256']==sha(S/'analyze.py') and s['verifier_source_sha256']==sha(S/'verify_public.py'))
        for rel,digest in s['public_sha256'].items():check('export '+rel,sha(P/rel)==digest)
        plot=read(P/'plot-provenance.json');check('figureprovenance',all(sha(R/rel)==digest for rel,digest in plot['input_sha256'].items()) and all(sha(P/rel)==digest for rel,digest in plot['output_sha256'].items()))
    return dict(status='passed',checks=len(checks),full_literal_census_and_original_query_audit_inherited_by_hash=True,new_literal_enumerations=0,new_oracle_labels=0,all1541_independent_graph_filters_and_exact_prediction_partition_verified=True,all1408_initial_positive_filters_and_ceiling_intersections_verified=True,all_directed_pair_counts_and_lexwitnesses_independently_bitwise_verified=len(pairs)*2,known704successes_separated_from704new_failure_diagnostics=True,elapsed_seconds=time.perf_counter()-start)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4]);print(json.dumps(verify(parser.parse_args().root)))


