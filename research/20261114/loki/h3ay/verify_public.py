"""Verify actual alias separation, complete populations and exact witness probabilities."""
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
    start=time.perf_counter();R=Path(root);S=R/'research/20261114/loki/h3ay';P=R/'results/research/loki/h3ay';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();checks=[]
    def check(name,condition):assert condition,name;checks.append(name)
    p=read(S/'protocol.json');g=read(P/'capacity-gate.json');raw=read(P/'study-result.json')
    check('completewithinallcaps',g['status']==raw['status']=='complete_verified' and not g['monitor_expired'] and g['aggregate_cpu_seconds']<=120 and g['wall_seconds']<=180 and p['budget']['GPUs']==0 and p['budget']['N13_supports']==2)
    check('frozenprotocolandreceipts',p['frozen_utc']<raw['utc'] and sha(S/'protocol.json')==g['protocol_sha256'] and g['source_sha256']==sha(S/'study.py') and all(g[k]==raw[k] for k in raw if k!='utc'))
    for rel,digest in p['source_sha256'].items():check('source '+rel,sha(R/rel)==digest)
    check('fixedA/Bpairandcriterion',p['families']==[dict(name='A',orbits=[[1,2],[3,4]]),dict(name='B',orbits=[[1,2],[2,7]])] and p['primary']['positive_labels']==16 and p['primary']['probability']==[19,20] and p['primary']['both_directed_differences_nonempty'] and p['primary']['both_families'])
    populations={};graphs={};censuses=[]
    with tempfile.TemporaryDirectory(prefix='h3ay-public-') as tmp:
        tmp=Path(tmp);binary=tmp/'literal';subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-Wall','-Wextra','-Werror',str(S/'reverse_literal.cpp'),'-o',str(binary)],check=True)
        def enumerate_F(F,n,label):
            inp=tmp/f'{label}.bin';out=tmp/f'{label}-columns.bin';F.astype(np.uint8).tofile(inp);r=subprocess.run([str(binary),str(n),str(inp),str(out)],check=True,capture_output=True,text=True);columns=np.fromfile(out,np.uint8).reshape(-1,n);assert len(columns)==int(r.stdout);return columns
        for n in (12,13):
            ordinary=enumerate_F(graph(n,[]),n,f'ordinary{n}');check(f'N{n}completeordinarypopulation',len(ordinary)==(14200 if n==12 else 73712))
            for cfg in p['families']:
                name=cfg['name'];rules=[tuple(o) for o in cfg['orbits']];F=graph(n,rules);columns=enumerate_F(F,n,f'{name}{n}');filtered=ordinary[~witnesses(ordinary,rules).any(1)];check(f'{name}N{n}fullcoordinategraphandcompletepopulation',np.array_equal(F,np.load(P/name/f'N{n}-F.npy')) and np.array_equal(columns,np.load(P/name/f'N{n}-columns.npy')) and np.array_equal(columns,filtered));populations[name,n]=columns;graphs[name,n]=F
                if n==13:
                    h=histogram(columns);keys=np.flatnonzero(h);census=np.load(P/name/'N13-clue-census.npz');check(f'{name}N13fullintegercluecensus',np.array_equal(keys,census['keys']) and np.array_equal(h[keys],census['counts']));censuses.append(dict(family=name,states=len(columns),distinct_keys=len(keys),total_occurrences=377*len(columns),sha256=sha(P/name/'N13-clue-census.npz')))
        A12=populations['A',12];B12=populations['B',12];check('N12completeobservationalalias',np.array_equal(A12,B12) and len(A12)==120 and np.array_equal(A12,np.load(P/'N12-shared-columns.npy')) and g['N12_populations_identical'] and g['N12_shared_states']==120 and g['N12_shared_sha256']==sha(P/'N12-shared-columns.npy'))
        A=populations['A',13];B=populations['B',13];As=set(map(bytes,A));Bs=set(map(bytes,B));inter=np.array([r for r in A if bytes(r) in Bs],np.uint8).reshape(-1,13);check('completeN13intersection',np.array_equal(inter,np.load(P/'N13-intersection.npy')) and len(inter)==g['N13_intersection_states'] and g['N13_intersection_sha256']==sha(P/'N13-intersection.npy'));rows=[];witnesses_out=[];symmetric=0
        for name,other,boards,otherkeys in [('A','B',A,Bs),('B','A',B,As)]:
            difference=np.array([r for r in boards if bytes(r) not in otherkeys],np.uint8).reshape(-1,13);symmetric+=len(difference);check(f'{name}completeN13directeddifference',np.array_equal(difference,np.load(P/f'N13-{name}-only.npy')));M=len(boards);I=len(inter);eligible=M>=16;total=math.comb(M,16) if eligible else None;fail=math.comb(I,16) if eligible else None;success=total-fail if eligible else None;passed=eligible and len(difference)>0 and 20*success>=19*total
            # Independent hypergeometric split: sum subsets with >=1 exclusive board.
            if eligible:check(f'{name}independenthypergeometriccount',success==sum(math.comb(len(difference),j)*math.comb(I,16-j) for j in range(1,17)))
            row=dict(true_family=name,other_family=other,N13_population=M,intersection_states=I,directed_difference_states=len(difference),directed_difference_nonempty=len(difference)>0,dose16_eligible=eligible,total_subsets=str(total) if eligible else None,indistinguishable_subsets=str(fail) if eligible else None,distinguishing_subsets=str(success) if eligible else None,distinguishing_probability=success/total if eligible else None,distinguishing_pct=100*success/total if eligible else None,probability_at_least95=bool(eligible and 20*success>=19*total),decision='supported' if passed else 'falsified' if eligible else 'ineligible_population_fewer_than16',population_sha256=sha(P/name/'N13-columns.npy'),difference_sha256=sha(P/f'N13-{name}-only.npy'));rows.append(row)
            if len(difference):
                board=difference[0];cells=13*np.arange(13)+board;violations=[dict(row_a=int(a),col_a=int(board[a]),row_b=int(b),col_b=int(board[b]),absolute_displacement=[b-a,abs(int(board[b])-int(board[a]))]) for a,b in itertools.combinations(range(13),2) if graphs[other,13][cells[a],cells[b]]];check(f'{name}lexfirstwitnessvalidhereandinvalidother',bool(violations) and not any(graphs[name,13][cells[a],cells[b]] for a,b in itertools.combinations(range(13),2)));witnesses_out.append(dict(valid_under=name,invalid_under=other,columns_zero_based=board.tolist(),columns_one_based=(board+1).tolist(),selection='Lexicographically first full board in the directed set difference.',all_violating_occupied_pairs_under_other=violations))
        check('allwitnessrecordsandcensuses',read(P/'distinguishing-witnesses.json')==dict(witnesses=witnesses_out,none_if_empty_directed_difference=True) and g['witnesses_sha256']==sha(P/'distinguishing-witnesses.json') and g['clue_censuses']==censuses)
        decision='inconclusive_ineligible_population' if not all(r['dose16_eligible'] for r in rows) else 'supported' if all(r['decision']=='supported' for r in rows) else 'falsified';check('allrowsandfrozenconjunction',rows==g['rows'] and symmetric==g['N13_symmetric_difference_states'] and g['decision']==decision and g['no_new_observed_labels'])
    if require_summary:
        s=read(P/'summary.json');check('summaryidentities',s['status']=='validated' and all(s[k]==g[k] for k in ('decision','rows','N12_shared_states','N13_intersection_states','N13_symmetric_difference_states','clue_censuses')) and s['protocol_sha256']==sha(S/'protocol.json') and s['audit_source_sha256']==sha(S/'analyze.py') and s['verifier_source_sha256']==sha(S/'verify_public.py'))
        for rel,digest in s['public_sha256'].items():check('export '+rel,sha(P/rel)==digest)
        f=read(P/'plot-provenance.json');check('figureprovenance',all(sha(R/rel)==digest for rel,digest in f['input_sha256'].items()) and all(sha(P/rel)==digest for rel,digest in f['output_sha256'].items()))
    return dict(status='passed',checks=len(checks),bothN12andN13completepopulations_reenumerated=True,N12completeobservationalidentity_verified=True,N13intersection_and_both_directed_differences_verified=True,bothlexfirstconcretewitnesses_and_everyviolation_checked=True,exactprobabilities_independently_checked_by_hypergeometric_sum=True,fullintegercluecensuses_and_allprimarydecisions_verified=True,elapsed_seconds=time.perf_counter()-start)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4]);print(json.dumps(verify(parser.parse_args().root)))


