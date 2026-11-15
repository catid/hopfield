"""Independently verify fresh two-orbit banks, all source normalizers and exact graphs."""
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


def diagnostics(root,public):
    R=Path(root);P=Path(public);read=lambda p:json.loads(p.read_text());raw=read(P/'study-result.json');candidates=read(P/'candidate-normalizers.json')['candidates'];ordinary=np.load(R/'results/research/loki/h3at/source-ordinary-columns.npy');bits=np.load(R/'results/research/loki/h3at/candidate-validity-bits.npz')['bits'];details=[]
    for row in raw['rows']:
        if row['ML']['every_tie_exact_N13_graph'] and row['MAP']['every_tie_exact_N13_graph']:continue
        true=row['true_candidate'];records=[]
        for arm in ('ML','MAP'):
            for cid in row[arm]['selected_candidates']:
                t=np.unpackbits(bits[true],bitorder='little')[:len(ordinary)].astype(bool);s=np.unpackbits(bits[cid],bitorder='little')[:len(ordinary)].astype(bool);TF=graph(13,[tuple(o) for o in candidates[true]['orbits']]);F=graph(13,[tuple(o) for o in candidates[cid]['orbits']]);a,b=np.triu_indices(169,1)
                records.append(dict(arm=arm,candidate=cid,orbits=candidates[cid]['orbits'],source_population_identical=bool(np.array_equal(t,s)),source_true_only_boards=int((t&~s).sum()),source_selected_only_boards=int((s&~t).sum()),N13_graph_identical=bool(np.array_equal(TF,F)),N13_false_edges=int(((F[a,b]>0)&(TF[a,b]==0)).sum()),N13_missed_edges=int(((F[a,b]==0)&(TF[a,b]>0)).sum())))
        details.append(dict(family=row['family'],bank=row['bank'],true_candidate=true,true_orbits=candidates[true]['orbits'],selected=records))
    return dict(scope='Post-hoc descriptive explanations of preserved failed exact-graph banks; no capacity or target-state measurements.',all_failed_banks=details)


def verify(root,require_summary=True):
    import gzip
    from fractions import Fraction
    start=time.perf_counter();R=Path(root);S=R/'research/20261114/loki/h3ax';P=R/'results/research/loki/h3ax';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();checks=[]
    def check(name,value):assert value,name;checks.append(name)
    p=read(S/'protocol.json');g=read(P/'capacity-gate.json');raw=read(P/'study-result.json');norm=read(P/'candidate-normalizers.json');prior=read(R/p['normalizers']['selection_path']);packed=np.load(R/p['normalizers']['bits_path'])['bits']
    check('complete within all frozen caps',g['status']==raw['status']=='complete_verified' and not g['monitor_expired'] and g['aggregate_cpu_seconds']<=120 and g['wall_seconds']<=180 and len(g['rows'])<=180 and p['budget']['GPUs']==0)
    check('frozenprotocolandreceiptidentities',p['frozen_utc']<raw['utc'] and sha(S/'protocol.json')==g['protocol_sha256']==norm['protocol_sha256'] and g['source_sha256']==sha(S/'study.py') and g['candidate_normalizers_sha256']==sha(P/'candidate-normalizers.json') and all(g[k]==raw[k] for k in raw if k!='utc'))
    for category in ('source_sha256','source_data_sha256'):
        for rel,digest in p[category].items():check(category+' '+rel,sha(R/rel)==digest)
    small=list(itertools.combinations(range(1,6),2));families=[dict(index=i,true_orbits=[list(a),list(b)],seeds=[2026116000+4*i+j for j in range(4)]) for i,(a,b) in enumerate(itertools.combinations(small,2))];orbits=list(itertools.combinations(range(1,12),2));choices=[chosen for k in (0,1,2) for chosen in itertools.combinations(range(55),k)]
    check('fixed45familygridandall180freshseeds',p['families']==families and p['candidate_prior']['orbits']==[list(o) for o in orbits] and len({seed for f in families for seed in f['seeds']})==180 and len(norm['candidates'])==1541)
    check('exacthierarchicalprior',sum(Fraction(1,3*math.comb(55,len(chosen))) for chosen in choices)==1)
    population=[];targetgraphs=[];rows=[];results=[]
    with tempfile.TemporaryDirectory(prefix='h3ax-public-') as tmp:
        tmp=Path(tmp);binary=tmp/'literal';subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-Wall','-Wextra','-Werror',str(S/'reverse_literal.cpp'),'-o',str(binary)],check=True)
        def enumerate_F(F,label):
            inp=tmp/f'{label}.bin';out=tmp/f'{label}-columns.bin';F.astype(np.uint8).tofile(inp);r=subprocess.run([str(binary),'12',str(inp),str(out)],check=True,capture_output=True,text=True);columns=np.fromfile(out,np.uint8).reshape(-1,12);assert len(columns)==int(r.stdout);return columns
        ordinary=enumerate_F(graph(12,[]),'ordinary');check('completeordinarysource',ordinary.shape==(14200,12) and np.array_equal(ordinary,np.load(R/p['normalizers']['ordinary_path'])));incidence=witnesses(ordinary,orbits)
        for cid,chosen in enumerate(choices):
            rules=[orbits[j] for j in chosen];F=graph(12,rules);columns=enumerate_F(F,f'c{cid}');mask=~incidence[:,chosen].any(1) if chosen else np.ones(len(ordinary),bool);M=len(columns);check(f'candidate{cid}allliteralpopulationandpackedbits',np.array_equal(columns,ordinary[mask]) and np.array_equal(np.packbits(mask,bitorder='little'),packed[cid]));base=dict(candidate=cid,orbit_indices=list(chosen),orbits=[list(o) for o in rules],complete_source_cardinality=M,source_columns_bytes_sha256=hashlib.sha256(columns.tobytes()).hexdigest(),source_F_bytes_sha256=hashlib.sha256(F.tobytes()).hexdigest());check(f'candidate{cid}inheritedmetadata',base==prior['candidates'][cid]);denom=math.comb(M,16) if M>=16 else None;check(f'candidate{cid}exactlikelihoodandMAPscore',norm['candidates'][cid]==dict(base,orbit_count=len(chosen),models_in_class=math.comb(55,len(chosen)),likelihood_denominator_if_eligible=str(denom) if denom is not None else None,MAP_integer_score_without_common3=str(math.comb(55,len(chosen))*denom) if denom is not None else None));population.append(columns);targetgraphs.append(graph(13,rules))
        check('all1541literalN13graphsaredistinct',len({F.tobytes() for F in targetgraphs})==1541 and norm['all1541_N13_graphs_distinct'])
        for cfg in families:
            family=cfg['index'];chosen=tuple(sorted(orbits.index(tuple(o)) for o in cfg['true_orbits']));true=choices.index(chosen);columns=population[true];M=len(columns);eligible=M>=16;folder=P/f'f{family}';check(f'family{family}completepopulationandliteraltruthgraph',np.array_equal(columns,np.load(folder/'source-population.npy')) and np.array_equal(targetgraphs[true],np.load(folder/'true-N13-F.npy')));familyrows=[]
            if eligible:
                for bank,seed in enumerate(cfg['seeds']):
                    out=folder/f'b{bank}';draw=np.load(out/'draw.npz');indices=np.random.default_rng(seed).choice(M,16,replace=False);selected=columns[indices];seen=witnesses(selected,orbits);check(f'family{family}bank{bank}freshuniformdrawandwitnesses',np.array_equal(indices,draw['indices']) and np.array_equal(selected,draw['columns']) and np.array_equal(seen,draw['orbit_incidence']) and len(set(map(bytes,selected)))==16);scorebytes=gzip.decompress((out/'scores.json.gz').read_bytes());record=json.loads(scorebytes);scores=[]
                    for cid,candidate in enumerate(choices):
                        violations=int(seen[:,candidate].any(1).sum()) if candidate else 0;consistent=violations==0;count=len(population[cid]);check(f'f{family}b{bank}c{cid}samplecardinality',not consistent or count>=16);scores.append(dict(candidate=cid,sample_violations=violations,consistent=consistent,likelihood_denominator=str(math.comb(count,16)) if consistent else None,MAP_integer_score_without_common3=str(math.comb(55,len(candidate))*math.comb(count,16)) if consistent else None))
                    selection={}
                    for arm,key in [('ML','likelihood_denominator'),('MAP','MAP_integer_score_without_common3')]:
                        best=min(int(r[key]) for r in scores if r['consistent']);winners=[r['candidate'] for r in scores if r['consistent'] and int(r[key])==best];selection[arm]=dict(selected_candidates=winners,best_integer_score=str(best),unique=len(winners)==1,every_tie_exact_N13_graph=all(np.array_equal(targetgraphs[cid],targetgraphs[true]) for cid in winners),selected_orbit_counts=[len(choices[cid]) for cid in winners],contains_true_candidate=true in winners)
                    check(f'family{family}bank{bank}all1541scoresandallMLMAPties',record==dict(family=family,bank=bank,seed=seed,scores=scores,selections=selection));row=dict(family=family,bank=bank,seed=seed,population_size=M,true_candidate=true,ML=selection['ML'],MAP=selection['MAP'],draw_sha256=sha(out/'draw.npz'),scores_sha256=hashlib.sha256(scorebytes).hexdigest());check(f'family{family}bank{bank}rawoutcome',row==g['rows'][len(rows)]);rows.append(row);familyrows.append(row)
                ML=sum(r['ML']['every_tie_exact_N13_graph'] for r in familyrows);MAP=sum(r['MAP']['every_tie_exact_N13_graph'] for r in familyrows);decision='supported' if MAP>=3 and MAP>=ML else 'falsified'
            else:
                ML=MAP=None;decision='ineligible_source_fewer_than16';check(f'family{family}ineligiblehasnodraws',not any(folder.glob('b*')))
            expected=dict(family=family,true_orbits=cfg['true_orbits'],true_candidate=true,source_population_size=M,eligible=eligible,drawn_banks=len(familyrows),ML_exact_banks=ML,MAP_exact_banks=MAP,required_MAP_exact_banks=3,requires_MAP_at_least_ML=True,decision=decision,source_population_sha256=sha(folder/'source-population.npy'),true_N13_F_sha256=sha(folder/'true-N13-F.npy'));check(f'family{family}eligibilityandprimarydecision',expected==g['family_results'][family]);results.append(expected)
        eligible=[f for f in results if f['eligible']];decision='inconclusive_no_eligible_families' if not eligible else 'supported' if all(f['decision']=='supported' for f in eligible) else 'falsified';check('all45familiesandeligiblebankconjunction',g['decision']==decision and g['eligible_families']==len(eligible) and g['ineligible_families']==45-len(eligible) and g['fixed_grid_families']==45 and g['new_positive_draws']==len(rows)==4*len(eligible) and g['all_ties_retained'] and g['no_target_capacity_or_state_enumeration'])
    check('descriptivefailureexplanationsincludingN12observationalequivalence',read(P/'failure-diagnostics.json')==diagnostics(R,P))
    if require_summary:
        s=read(P/'summary.json');check('summaryidentities',s['status']=='validated' and all(s[k]==g[k] for k in ('decision','rows','family_results','eligible_families','ineligible_families','new_positive_draws')) and s['protocol_sha256']==sha(S/'protocol.json') and s['audit_source_sha256']==sha(S/'analyze.py') and s['verifier_source_sha256']==sha(S/'verify_public.py'))
        for rel,digest in s['public_sha256'].items():check('export '+rel,sha(P/rel)==digest)
        plot=read(P/'plot-provenance.json');check('figureprovenance',all(sha(R/rel)==digest for rel,digest in plot['input_sha256'].items()) and all(sha(P/rel)==digest for rel,digest in plot['output_sha256'].items()))
    return dict(status='passed',checks=len(checks),all1541_complete_sourcepopulations_independently_reenumerated=True,all45_truefamilyeligibilities_andgraphs_checked=True,all100_freshdraws_and154100_scores_replayed=True,all_ML_MAP_ties_and_every_selected_graph_verified=True,all_primarydecisions_and_failure_diagnostics_verified=True,elapsed_seconds=time.perf_counter()-start)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4]);print(json.dumps(verify(parser.parse_args().root)))


