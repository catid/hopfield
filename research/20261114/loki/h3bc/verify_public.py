"""Independent full-class population reconstruction and sealed trajectory replay."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import tempfile
import time
import numpy as np


def edge_parts(n,orbits):
    base=np.zeros((n*n,n*n),np.int32);extra=np.zeros((len(orbits),n*n,n*n),np.int32);lookup={o:j for j,o in enumerate(orbits)}
    for a,b in itertools.combinations(range(n*n),2):
        dr=abs(a//n-b//n);dc=abs(a%n-b%n)
        if dr==0 or dc==0 or dr==dc:base[a,b]=base[b,a]=1
        key=tuple(sorted((dr,dc)))
        if key in lookup:extra[lookup[key],a,b]=extra[lookup[key],b,a]=1
    return base,extra


def witnesses(boards,orbits):
    lookup={o:j for j,o in enumerate(orbits)};hit=np.zeros((len(boards),len(orbits)),bool)
    for i,board in enumerate(boards):
        for a,b in itertools.combinations(range(boards.shape[1]),2):
            key=tuple(sorted((b-a,abs(int(board[b])-int(board[a])))))
            if key in lookup:hit[i,lookup[key]]=True
    return hit


def verify(root,require_summary=True):
    start=time.perf_counter();R=Path(root);S=R/'research/20261114/loki/h3bc';P=R/'results/research/loki/h3bc';read=lambda path:json.loads(path.read_text());sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest();checks=[]
    def check(name,value):assert value,name;checks.append(name)
    p=read(S/'protocol.json');g=read(P/'capacity-gate.json');raw=read(P/'study-result.json');prep=read(P/'preparation.json');old=read(R/p['normalizers']['selection_path']);oldgrid=read(R/'results/research/loki/h3ax/summary.json');sourcebits=np.load(R/p['normalizers']['bits_path'])['bits']
    check('wholebudgetandcomplete',g['status']==raw['status']=='complete_verified' and not g['monitor_expired'] and g['aggregate_cpu_seconds']<=600 and g['wall_seconds']<=900 and p['budget']['GPUs']==0)
    check('freezeandstagereceipts',p['frozen_utc']<prep['utc']<raw['utc'] and sha(S/'protocol.json')==g['protocol_sha256']==prep['protocol_sha256'] and sha(S/'study.py')==g['source_sha256'] and sha(P/'preparation.json')==g['preparation_sha256'] and all(g[k]==raw[k] for k in raw if k!='utc'))
    for category in ['source_sha256','source_data_sha256']:
        for rel,digest in p[category].items():check(category+' '+rel,sha(R/rel)==digest)
    orbits=list(itertools.combinations(range(1,12),2));choices=[x for k in [0,1,2] for x in itertools.combinations(range(55),k)];ids=list(range(1541));expectedcases=[]
    for i in range(8):
        true=[(2,3),(2,5)][i//4];expectedcases.append(dict(case=len(expectedcases),archive='H3ao',family=i//4,bank=i%4,seed=2026113400+i,true_candidate=choices.index((orbits.index(true),)),true_orbits=[list(true)],draw_path=f'results/research/loki/h3at/b{i}/source-evidence.npz',draw_key='sample_columns'))
    for row in oldgrid['rows']:
        family=row['family'];bank=row['bank'];expectedcases.append(dict(case=len(expectedcases),archive='H3ax',family=family,bank=bank,seed=row['seed'],true_candidate=row['true_candidate'],true_orbits=oldgrid['family_results'][family]['true_orbits'],draw_path=f'results/research/loki/h3ax/f{family}/b{bank}/draw.npz',draw_key='columns'))
    check('fixed108cases1541modelsandfulltruthgrid',p['cases']==expectedcases and len(expectedcases)==108 and p['candidate_prior']['orbits']==[list(o) for o in orbits] and prep['candidate_ids']==ids and p['inherited_truth_grid']==g['inherited_truth_grid']==oldgrid['family_results'] and sum(not f['eligible'] for f in g['inherited_truth_grid'])==20)
    targetbits=np.load(P/'predictions-bits.npy');population_rows=[];sourcepop={};sourcepred=[]
    with tempfile.TemporaryDirectory(prefix='h3bc-public-') as tmp:
        tmp=Path(tmp);binary=tmp/'literal';subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-Wall','-Wextra','-Werror',str(S/'reverse_literal.cpp'),'-o',str(binary)],check=True)
        def enumerate_F(n,F):
            inp=tmp/'F.bin';out=tmp/'columns.bin';F.astype(np.uint8).tofile(inp);r=subprocess.run([str(binary),str(n),str(inp),str(out)],capture_output=True,text=True,check=True);columns=np.fromfile(out,np.uint8).reshape(-1,n);assert len(columns)==int(r.stdout);return columns
        base12,extra12=edge_parts(12,orbits);base13,extra13=edge_parts(13,orbits);ordinary12=enumerate_F(12,base12);boards=enumerate_F(13,base13)
        check('completeordinaryuniverses',ordinary12.shape==(14200,12) and boards.shape==(73712,13) and np.array_equal(ordinary12,np.load(R/p['normalizers']['ordinary_path'])) and np.array_equal(boards,np.load(P/'ordinary13.npy')) and all(bytes(a)<bytes(b) for a,b in zip(boards,boards[1:])))
        hit12=witnesses(ordinary12,orbits);hit13=witnesses(boards,orbits)
        for cid,chosen in enumerate(choices):
            m12=~hit12[:,chosen].any(1) if chosen else np.ones(len(ordinary12),bool);F12=base12.copy();F13=base13.copy()
            for j in chosen:F12|=extra12[j];F13|=extra13[j]
            c12=ordinary12[m12];sourcepop[cid]=c12;sourcepred.append(m12);record=dict(candidate=cid,orbit_indices=list(chosen),orbits=[list(orbits[j]) for j in chosen],complete_source_cardinality=len(c12),source_columns_bytes_sha256=hashlib.sha256(c12.tobytes()).hexdigest(),source_F_bytes_sha256=hashlib.sha256(F12.tobytes()).hexdigest())
            check(f'candidate{cid}sourcefullmaskandmetadata',np.array_equal(np.packbits(m12,bitorder='little'),sourcebits[cid]) and record==old['candidates'][cid])
            m13=~hit13[:,chosen].any(1) if chosen else np.ones(len(boards),bool);columns=enumerate_F(13,F13);packed=np.packbits(m13,bitorder='little')
            check(f'candidate{cid}targetliteralfilterandpacked',np.array_equal(columns,boards[m13]) and np.array_equal(packed,targetbits[cid]))
            population_rows.append(dict(candidate=cid,states=len(columns),columns_bytes_sha256=hashlib.sha256(columns.tobytes()).hexdigest(),F_bytes_sha256=hashlib.sha256(F13.tobytes()).hexdigest(),packed_bytes_sha256=hashlib.sha256(packed.tobytes()).hexdigest()))
        for cfg in g['inherited_truth_grid']:check(f'gridfamily{cfg["family"]}eligibility',len(sourcepop[cfg['true_candidate']])==cfg['source_population_size'] and cfg['eligible']==(cfg['source_population_size']>=16))
    sourcepred=np.array(sourcepred,bool);predictions=np.unpackbits(targetbits,axis=1,count=73712,bitorder='little').astype(bool)
    check('preparationandoptimizationparity',prep['populations']==population_rows and prep['ordinary_sha256']==sha(P/'ordinary13.npy') and prep['predictions_sha256']==sha(P/'predictions-bits.npy') and prep['unpack_benchmark']['exact_parity'] and np.array_equal(predictions[:17],np.unpackbits(targetbits[:17],axis=1,count=73712,bitorder='little').astype(bool)))
    ordinary_lookup={bytes(row):i for i,row in enumerate(ordinary12)};results=[];queries=0;same_state={};unordered_samples={}
    for cfg in expectedcases:
        case=cfg['case'];truth=cfg['true_candidate'];folder=P/f'cases/case{case}';sample=np.load(folder/'source-columns.npy');archive=np.load(R/cfg['draw_path'])[cfg['draw_key']];M=len(sourcepop[truth]);draw=np.random.default_rng(cfg['seed']).choice(M,64 if cfg['archive']=='H3ao' else 16,replace=False)[:16]
        check(f'case{case}rawpositivesandseed',sample.shape==(16,12) and len(set(map(bytes,sample)))==16 and np.array_equal(sample,archive) and np.array_equal(sample,sourcepop[truth][draw]))
        sampleids=[ordinary_lookup[bytes(row)] for row in sample];initial=np.flatnonzero(sourcepred[:,sampleids].all(1)).tolist();candidates=list(initial);source=read(folder/'source-evidence.json')
        check(f'case{case}all1541positiveonlyfilter',source['initial_candidates']==initial and truth in initial and source['case']==case and source['source_path']==cfg['draw_path'] and source['source_sha256']==sha(R/cfg['draw_path']) and source['source_columns_sha256']==sha(folder/'source-columns.npy') and source['positive_count']==16 and source['protocol_sha256']==sha(S/'protocol.json') and prep['utc']<source['utc'])
        key=(truth,tuple(sorted(map(bytes,sample))));unordered_samples.setdefault(key,[]).append(case);steps=[];queried=[]
        for step in range(4):
            if len(candidates)==1:break
            qfile=folder/f'q{step}-sealed.json';afile=folder/f'q{step}-answer.json';q=read(qfile);a=read(afile);request=dict(bits_path='predictions-bits.npy',candidate_ids=ids,board_count=len(boards),candidates=candidates,queried=queried)
            check(f'case{case}step{step}onlyallowedselectorinputs',q['request']==request)
            counts=np.count_nonzero(predictions[candidates],axis=0);squared=(len(candidates)-2*counts)**2;squared[queried]=10**12;index=int(np.argmin(squared));selection=dict(query_index=index,imbalance=int(abs(len(candidates)-2*counts[index])),optimal_board_count=int(np.count_nonzero(squared==squared[index])),candidate_predictions={str(cid):bool(predictions[cid,index]) for cid in candidates})
            check(f'case{case}step{step}globalbalanceandalllexties',q['selection']==selection and q['board_zero_based']==boards[index].tolist() and q['step']==step and q['protocol_sha256']==sha(S/'protocol.json') and index not in queried)
            skey=(tuple(candidates),tuple(queried));check(f'case{case}step{step}truthblindpolicy',skey not in same_state or same_state[skey]==index);same_state[skey]=index
            label=bool(predictions[truth,index]);remaining=[cid for cid in candidates if bool(predictions[cid,index])==label]
            check(f'case{case}step{step}sealbeforeoracle',source['utc']<q['utc']<=a['utc'] and q['monotonic_ns']<a['monotonic_ns'] and a['query_sha256']==sha(qfile))
            check(f'case{case}step{step}labelandallcompatiblemodels',a['label']==label and a['candidates_before']==candidates and a['candidates_after']==remaining and truth in remaining)
            steps.append(dict(step=step,query_index=index,label=label,candidates_before=candidates,candidates_after=remaining,query_sha256=sha(qfile),answer_sha256=sha(afile)));candidates=remaining;queried=queried+[index];queries+=1
        check(f'case{case}noextraqueries',len(list(folder.glob('q*-sealed.json')))==len(steps) and len(list(folder.glob('q*-answer.json')))==len(steps))
        result=dict(case=case,archive=cfg['archive'],family=cfg['family'],bank=cfg['bank'],seed=cfg['seed'],true_candidate=truth,initial_candidates=initial,remaining_candidates=candidates,exact_true_singleton=candidates==[truth],membership_labels_used=len(steps),positive_membership_labels=sum(s['label'] for s in steps),negative_membership_labels=sum(not s['label'] for s in steps),source_positive_labels=16,total_labels=16+len(steps),source_sha256=sha(folder/'source-evidence.json'),steps=steps)
        check(f'case{case}allretainedresults',result==read(folder/'result.json'));results.append(result)
    decision='supported' if all(r['exact_true_singleton'] for r in results) else 'falsified'
    check('all108outcomesandprimary',g['results']==results and g['decision']==decision and g['identified']==sum(r['exact_true_singleton'] for r in results) and g['membership_labels']==queries)
    duplicate_groups=[cases for cases in unordered_samples.values() if len(cases)>1]
    for group in duplicate_groups:check(f'duplicatebankgroup{group[0]}allretained',all([(x['query_index'],x['label'],x['candidates_after']) for x in results[case]['steps']]==[(x['query_index'],x['label'],x['candidates_after']) for x in results[group[0]]['steps']] for case in group))
    if require_summary:
        s=read(P/'summary.json');check('summaryidentities',s['status']=='validated' and all(s[k]==g[k] for k in ['decision','results','inherited_truth_grid','identified','membership_labels']) and s['protocol_sha256']==sha(S/'protocol.json') and s['audit_source_sha256']==sha(S/'analyze.py') and s['verifier_source_sha256']==sha(S/'verify_public.py'))
        for rel,digest in s['public_sha256'].items():check('export '+rel,sha(P/rel)==digest)
        plot=read(P/'plot-provenance.json');check('figureprovenance',all(sha(R/rel)==digest for rel,digest in plot['input_sha256'].items()) and all(sha(P/rel)==digest for rel,digest in plot['output_sha256'].items()))
    return dict(status='passed',checks=len(checks),all1541N12_masks_and_N13_literal_populations_verified=True,all108source_seeds_and_full_candidate_filters_verified=True,all_sealed_queries_replayed=queries,all20_ineligible_grid_families_retained=True,duplicate_unordered_source_bank_groups=duplicate_groups,elapsed_seconds=time.perf_counter()-start)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4]);print(json.dumps(verify(parser.parse_args().root)))
