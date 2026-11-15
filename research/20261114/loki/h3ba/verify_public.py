"""Audit the full source alias partition and all bounded next-size populations."""
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time
import numpy as np


def graph(n, orbits):
    F = np.zeros((n*n, n*n), np.int32)
    orbit_set = {tuple(sorted(o)) for o in orbits}
    for u in range(n*n):
        for v in range(u+1, n*n):
            a, b = abs(u//n-v//n), abs(u%n-v%n)
            F[u, v] = F[v, u] = a == 0 or b == 0 or a == b or tuple(sorted((a, b))) in orbit_set
    return F


def witnesses(columns, orbits):
    lookup = {tuple(o): j for j, o in enumerate(orbits)}
    hits = np.zeros((len(columns), len(orbits)), bool)
    for k, board in enumerate(columns):
        for a, b in itertools.combinations(range(len(board)), 2):
            orbit = tuple(sorted((b-a, abs(int(board[b])-int(board[a])))))
            if orbit in lookup:
                hits[k, lookup[orbit]] = True
    return hits


def histogram(columns):
    cells = 13*np.arange(13)[None, :] + columns.astype(np.int64)
    counts = np.zeros(170**3, np.int64)
    for width in (1, 2, 3):
        subsets = np.array(list(itertools.combinations(range(13), width)))
        keys = ((cells[:, subsets]+1)*np.array([170**(2-j) for j in range(width)])).sum(-1)
        counts += np.bincount(keys.ravel(), minlength=len(counts))
    assert counts.sum() == 377*len(columns)
    return counts


def verify(root,require_summary=True):
    start=time.perf_counter();R=Path(root);S=R/'research/20261114/loki/h3ba';P=R/'results/research/loki/h3ba';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();checks=[]
    def check(name,value):assert value,name;checks.append(name)
    p=read(S/'protocol.json');g=read(P/'capacity-gate.json');raw=read(P/'study-result.json');partition=read(P/'N12-partition.json');pooled=read(P/'pooled-N12-N13-partition.json');comparison=read(P/'N14-behavior-comparison.json');old=read(R/p['normalizers']['selection_path']);bits=np.load(R/p['normalizers']['bits_path'])['bits']
    check('completedwithincaps',g['status']==raw['status']=='complete_verified' and not g['monitor_expired'] and g['aggregate_cpu_seconds']<=600 and g['wall_seconds']<=900 and g['N13_models']<=17 and g['N14_models']<=17 and p['budget']['GPUs']==0)
    check('protocolandstagedreceipts',p['frozen_utc']<partition['utc']<pooled['utc']<raw['utc'] and sha(S/'protocol.json')==partition['protocol_sha256']==g['protocol_sha256'] and g['source_sha256']==sha(S/'study.py') and all(g[k]==raw[k] for k in raw if k!='utc') and g['N12_partition_sha256']==pooled['N12_partition_sha256']==sha(P/'N12-partition.json') and g['pooled_partition_sha256']==sha(P/'pooled-N12-N13-partition.json') and g['behavior_comparison_sha256']==sha(P/'N14-behavior-comparison.json'))
    for category in ('source_sha256','source_data_sha256'):
        for rel,digest in p[category].items():check(category+' '+rel,sha(R/rel)==digest)
    orbits=list(itertools.combinations(range(1,12),2));choices=[chosen for k in (0,1,2) for chosen in itertools.combinations(range(55),k)];check('fixed1541modelscope',p['candidate_prior']['orbits']==[list(o) for o in orbits] and len(partition['candidates'])==1541)
    populations={};classes=[];lookup={};modelclass={};candidates=[];stages={};results=[]
    with tempfile.TemporaryDirectory(prefix='h3ba-public-') as tmp:
        tmp=Path(tmp);binary=tmp/'literal';subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-Wall','-Wextra','-Werror',str(S/'reverse_literal.cpp'),'-o',str(binary)],check=True)
        def enumerate_F(F,n,label):
            inp=tmp/f'{label}.bin';out=tmp/f'{label}-columns.bin';F.astype(np.uint8).tofile(inp);r=subprocess.run([str(binary),str(n),str(inp),str(out)],capture_output=True,text=True,check=True);columns=np.fromfile(out,np.uint8).reshape(-1,n);assert len(columns)==int(r.stdout);return columns
        ordinary=enumerate_F(graph(12,[]),12,'ordinary12');check('ordinaryN12complete',ordinary.shape==(14200,12) and np.array_equal(ordinary,np.load(R/p['normalizers']['ordinary_path'])));hit=witnesses(ordinary,orbits)
        for cid,chosen in enumerate(choices):
            rules=[orbits[j] for j in chosen];F=graph(12,rules);columns=enumerate_F(F,12,f'c{cid}');mask=~hit[:,chosen].any(1) if chosen else np.ones(len(ordinary),bool);check(f'candidate{cid}completepopulation',np.array_equal(columns,ordinary[mask]) and np.array_equal(np.packbits(mask,bitorder='little'),bits[cid]));record=dict(candidate=cid,orbit_indices=list(chosen),orbits=[list(o) for o in rules],complete_source_cardinality=len(columns),source_columns_bytes_sha256=hashlib.sha256(columns.tobytes()).hexdigest(),source_F_bytes_sha256=hashlib.sha256(F.tobytes()).hexdigest());check(f'candidate{cid}allmetadata',record==old['candidates'][cid]==partition['candidates'][cid]);candidates.append(record);key=columns.tobytes()
            if key not in lookup:lookup[key]=len(classes);classes.append(dict(class_id=len(classes),source_population_size=len(columns),eligible_M_at_least16=len(columns)>=16,members=[]))
            group=lookup[key];classes[group]['members'].append(cid);modelclass[cid]=group
        check('all1541wholepopulationequivalenceclasses',classes==partition['classes'])
        families=[dict(family=cid,true_orbits=c['orbits'],true_candidate=cid,N12_population=c['complete_source_cardinality'],eligible=c['complete_source_cardinality']>=16,N12_class=modelclass[cid],N12_alias_members=classes[modelclass[cid]]['members']) for cid,c in enumerate(candidates)]
        previous=read(R/p['archive_partition']);check('exactpriorpartitionreusedwithfullreverification',previous['classes']==classes and previous['candidates']==candidates and partition['previous_partition_sha256']==sha(R/p['archive_partition']))
        relevant=sorted({r['N12_class'] for r in families if r['eligible'] and len(r['N12_alias_members'])>1});N13models=sorted({cid for group in relevant for cid in classes[group]['members']});eligibleclasses=[c for c in classes if c['eligible_M_at_least16']];descriptive=dict(total_models=1541,total_equivalence_classes=len(classes),eligible_models=sum(len(c['members']) for c in eligibleclasses),eligible_classes=len(eligibleclasses),eligible_aliased_models=sum(len(c['members']) for c in eligibleclasses if len(c['members'])>1),eligible_nontrivial_classes=sum(len(c['members'])>1 for c in eligibleclasses));check('allfamiliesprevalenceandfullrelevantscope',families==partition['families'] and relevant==partition['relevant_nontrivial_classes'] and N13models==partition['N13_model_ids'] and descriptive==partition['descriptive_all1541']==g['descriptive_all1541'])
        def stage(n,models):
            if not models:return []
            universe=enumerate_F(graph(n,[]),n,f'ordinary{n}');check(f'N{n}completeordinary',len(universe)=={13:73712,14:365596}[n]);records=[]
            for cid in models:
                rules=[orbits[j] for j in choices[cid]];F=graph(n,rules);columns=enumerate_F(F,n,f'N{n}c{cid}');filtered=universe[~witnesses(universe,rules).any(1)];folder=P/f'N{n}/candidate{cid}';check(f'N{n}candidate{cid}allpopulationandgraph',np.array_equal(columns,filtered) and np.array_equal(columns,np.load(folder/'columns.npy')) and np.array_equal(F,np.load(folder/'F.npy')));populations[cid,n]=columns;records.append(dict(candidate=cid,n=n,states=len(columns),columns_sha256=sha(folder/'columns.npy'),F_sha256=sha(folder/'F.npy')))
            return records
        stages['N13']=stage(13,N13models);refined=[];modelrefined={};N13witnesses=[]
        for group in relevant:
            for a,b in itertools.combinations(classes[group]['members'],2):
                A=populations[a,13];B=populations[b,13];As=set(map(bytes,A));Bs=set(map(bytes,B));directions=[]
                for left,right,pop,other in [(a,b,A,Bs),(b,a,B,As)]:
                    difference=[row for row in pop if bytes(row) not in other];directions.append(dict(valid_candidate=left,invalid_candidate=right,directed_difference_size=len(difference),lexfirst_columns_zero_based=difference[0].tolist() if difference else None))
                N13witnesses.append(dict(N12_class=group,candidates=[a,b],N13_populations_identical=bool(np.array_equal(A,B)),directions=directions))
        check('all28N13pairwisebehaviorcomparisonsandwitnesses',read(P/'N13-pair-comparison.json')==dict(pairs=N13witnesses,all_pairs_within_each_eligible_N12_class_retained=True) and g['N13_pair_comparison_sha256']==sha(P/'N13-pair-comparison.json'))
        for group in relevant:
            sub={}
            for cid in classes[group]['members']:sub.setdefault(populations[cid,13].tobytes(),[]).append(cid)
            for members in sub.values():
                record=dict(refined_class=len(refined),N12_class=group,N13_states=len(populations[members[0],13]),members=members);refined.append(record)
                for cid in members:modelrefined[cid]=record['refined_class']
        N14models=sorted({cid for r in refined if len(r['members'])>1 for cid in r['members']});check('exactpooledpartitionandN14scope',refined==pooled['refined_classes'] and N14models==pooled['N14_model_ids'] and stages['N13']==pooled['N13_stage_records']);stages['N14']=stage(14,N14models);behavior=[];witnesses_out=[]
        for record in refined:
            members=record['members']
            if len(members)==1:behavior.append(dict(refined_class=record['refined_class'],members=members,N14_measured=False,N14_behavior_identified=True,reason='Unique model after pooled complete evidence.'));continue
            sub={}
            for cid in members:sub.setdefault(populations[cid,14].tobytes(),[]).append(cid)
            for a,b in itertools.combinations(members,2):
                A=populations[a,14];B=populations[b,14];As=set(map(bytes,A));Bs=set(map(bytes,B))
                for left,right,pop,other in [(a,b,A,Bs),(b,a,B,As)]:
                    difference=[row for row in pop if bytes(row) not in other]
                    if difference:witnesses_out.append(dict(refined_class=record['refined_class'],valid_candidate=left,invalid_candidate=right,directed_difference_size=len(difference),lexfirst_columns_zero_based=difference[0].tolist()))
            behavior.append(dict(refined_class=record['refined_class'],members=members,N14_measured=True,N14_behavior_identified=len(sub)==1,N14_behavior_classes=list(sub.values()),N14_population_sizes={str(cid):len(populations[cid,14]) for cid in members},reason='All surviving model state populations compared exactly.'))
        check('allbehaviorclassesandwitnesses',comparison==dict(behavior_classes=behavior,witnesses=witnesses_out,all_directed_pair_differences_retained=True))
        for family in families:
            if not family['eligible']:result=dict(family,decision='ineligible_N12_population_fewer_than16',case='ineligible',pooled_alias_members=None,N14_measured=False)
            elif len(family['N12_alias_members'])==1:result=dict(family,decision='supported',case='N12_unique_model_vacuous',pooled_alias_members=family['N12_alias_members'],N14_measured=False)
            else:
                group=modelrefined[family['true_candidate']];record=behavior[group];members=record['members'];case='N13_unique_model_vacuous' if len(members)==1 else 'remaining_alias_N14_behavior_compared';result=dict(family,decision='supported' if record['N14_behavior_identified'] else 'falsified',case=case,pooled_alias_members=members,N14_measured=record['N14_measured'],refined_class=group)
            results.append(result)
        eligible=[r for r in results if r['eligible']];decision='inconclusive_no_eligible_truths' if not eligible else 'supported' if all(r['decision']=='supported' for r in eligible) else 'falsified';check('all1541modelcasesandprimarydecision',results==g['family_results'] and decision==g['decision'] and stages==g['stage_records'] and len(relevant)==g['relevant_nontrivial_N12_classes'] and len(N13models)==g['N13_models'] and len(N14models)==g['N14_models'] and len(refined)==g['pooled_refined_classes'])
        if not N14models:check('noN14measurementclaimorfiles',not(P/'N14').exists() and not any(r['N14_measured'] for r in results) and not stages['N14'])
    if require_summary:
        s=read(P/'summary.json');check('summaryidentities',s['status']=='validated' and all(s[k]==g[k] for k in ('decision','family_results','descriptive_all1541','relevant_nontrivial_N12_classes','N13_models','N14_models','pooled_refined_classes','stage_records')) and s['protocol_sha256']==sha(S/'protocol.json') and s['audit_source_sha256']==sha(S/'analyze.py') and s['verifier_source_sha256']==sha(S/'verify_public.py'))
        for rel,digest in s['public_sha256'].items():check('export '+rel,sha(P/rel)==digest)
        plot=read(P/'plot-provenance.json');check('figureprovenance',all(sha(R/rel)==digest for rel,digest in plot['input_sha256'].items()) and all(sha(P/rel)==digest for rel,digest in plot['output_sha256'].items()))
    return dict(status='passed',checks=len(checks),all1541_completeN12populations_and_equivalenceclasses_reenumerated=True,all1541eligibilities_and_scopechoices_verified=True,all_required_N13_N14_populations_and_refinements_verified=True,vacuous_cases_separated_from_actualN14measurements=True,all_primarydecisions_and_publicidentities_verified=True,elapsed_seconds=time.perf_counter()-start)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4]);print(json.dumps(verify(parser.parse_args().root)))


