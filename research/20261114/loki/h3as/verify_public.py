"""Independent literal source enumeration and direct inclusion-exclusion audit."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
from decimal import Decimal, localcontext
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time
import numpy as np


def graph(n, rules):
    F=np.zeros((n*n,n*n), np.uint8); rules={tuple(sorted(r)) for r in rules}
    for u,v in itertools.combinations(range(n*n),2):
        a,b=abs(u//n-v//n),abs(u%n-v%n)
        F[u,v]=F[v,u]=a==0 or b==0 or a==b or tuple(sorted((a,b))) in rules
    return F


def independent_witnesses(population, orbits):
    hit=np.zeros((len(population),len(orbits)),bool);lookup={o:i for i,o in enumerate(orbits)}
    for i,board in enumerate(population):
        for a,b in itertools.combinations(range(12),2):
            orbit=tuple(sorted((b-a,abs(int(board[b])-int(board[a])))))
            if orbit in lookup:hit[i,lookup[orbit]]=True
    return hit


def inclusion_exclusion(frequency, dose):
    # Direct submask summation, independent of transform update ordering and DP.
    contained=[sum(n for pattern,n in enumerate(frequency) if pattern&~mask==0) for mask in range(512)]
    below=[math.comb(n,dose) if n>=dose else 0 for n in contained];mass=[]
    for mask in range(512):
        sub=mask;value=0
        while True:
            value+=(-1)**((mask^sub).bit_count())*below[sub]
            if sub==0:break
            sub=(sub-1)&mask
        mass.append(value)
    return mass


def exact_presentation(rows):
    output=[]
    with localcontext() as context:
        context.prec=90
        for row in rows:
            success=int(row['successful_subsets']);total=int(row['total_subsets']);fail=total-success;gcd=math.gcd(fail,total)
            output.append(dict(family=row['family'],dose=row['dose'],failed_subsets=str(fail),failure_reduced_numerator=str(fail//gcd),failure_reduced_denominator=str(total//gcd),failure_probability_scientific=format(Decimal(fail)/Decimal(total),'.16E'),success_probability_decimal=str(Decimal(success)/Decimal(total)),success_pct_decimal=str(100*Decimal(success)/Decimal(total)),float_success_may_round_to_one=(success<total and row['unique_true_graph_probability']==1.0)))
    return output


def verify(root, require_summary=True):
    start=time.perf_counter();R=Path(root);S=R/'research/20261114/loki/h3as';P=R/'results/research/loki/h3as'
    read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();checks=[]
    def check(name,value):assert value,name;checks.append(name)
    p=read(S/'protocol.json');g=read(P/'capacity-gate.json');raw=read(P/'study-result.json');old=read(R/p['source_selection_path'])
    check('completed within frozen resource limits',g['status']==raw['status']=='complete_verified' and not g['monitor_expired'] and g['aggregate_cpu_seconds']<=120 and g['wall_seconds']<=180 and p['budget']['GPUs']==0)
    check('frozen protocol and result identities',p['frozen_utc']<raw['utc'] and sha(S/'protocol.json')==g['protocol_sha256'] and g['source_sha256']==sha(S/'study.py') and all(g[k]==raw[k] for k in raw if k!='utc'))
    for group in ('source_sha256','source_data_sha256'):
        for rel,digest in p[group].items():check(group+' '+rel,sha(R/rel)==digest)
    orbits=list(itertools.combinations(range(1,6),2));candidates=[chosen for width in (0,1,2) for chosen in itertools.combinations(range(10),width)]
    check('fixed prior and doses',p['candidate_prior']['orbits']==[list(o) for o in orbits] and p['doses']==[4,8,16,32,64] and p['primary']==dict(dose=16,probability_threshold=[99,100],all_families=True))
    check('small exhaustive algorithm parity receipt',g['validation']==dict(fixtures=5,dose_cases=45,all_literal_subsets_mobius_and_grouped_DP_equal=True))
    with tempfile.TemporaryDirectory(prefix='h3as-public-') as tmp:
        tmp=Path(tmp);binary=tmp/'literal'
        subprocess.run(['g++','-O3','-std=c++17','-DNDEBUG','-Wall','-Wextra','-Werror',str(S/'reverse_literal.cpp'),'-o',str(binary)],check=True)
        def enumerate_F(F,label):
            inp=tmp/f'{label}.bin';out=tmp/f'{label}columns.bin';F.tofile(inp)
            result=subprocess.run([str(binary),'12',str(inp),str(out)],capture_output=True,text=True,check=True);columns=np.fromfile(out,np.uint8).reshape(-1,12);assert len(columns)==int(result.stdout);return columns
        populations=[];sizes=[];graph_sets={12:set(),13:set()}
        for cid,chosen in enumerate(candidates):
            rules=[orbits[j] for j in chosen];F=graph(12,rules);columns=enumerate_F(F,f'candidate{cid}');populations.append(columns);sizes.append(len(columns))
            check(f'candidate{cid}literal complete source population and normalizer',np.array_equal(columns,np.load(R/f'results/research/loki/h3ar/candidate{cid}/source-columns.npy')) and len(columns)==old['candidates'][cid]['complete_source_cardinality'])
            graph_sets[12].add(F.tobytes());graph_sets[13].add(graph(13,rules).tobytes())
        check('all56graphs distinct at both sizes',all(len(v)==56 for v in graph_sets.values()) and g['graph_distinctness']=={str(n):dict(distinct_graphs=56,candidate_graphs=56) for n in (12,13)})
        rows=[]
        for family,cfg in enumerate(p['families']):
            folder=P/f'family{family}';d=read(folder/'exact-distribution.json');patterns=read(folder/'selection-patterns.json');true=orbits.index(tuple(cfg['leaper']));truecid=candidates.index((true,));population=populations[truecid];others=[j for j in range(10) if j!=true];hit=independent_witnesses(population,orbits)
            check(f'family{family}complete population and properness',np.array_equal(population,np.load(R/cfg['source_path'])) and np.array_equal(population,np.load(folder/'population.npy')) and len(population)==d['population_size']<sizes[0] and not hit[:,true].any() and d['true_candidate']==truecid and d['true_orbit_index']==true and d['other_orbit_indices']==others)
            masks=[sum(1<<bit for bit,j in enumerate(others) if hit[i,j]) for i in range(len(population))];freq=[masks.count(mask) for mask in range(512)]
            check(f'family{family}all board witnesses and512joint frequencies',np.array_equal(masks,np.load(folder/'witness-masks.npy')) and np.array_equal(freq,np.load(folder/'frequency.npy')))
            for name,key in [('population.npy','population_sha256'),('witness-masks.npy','witness_masks_sha256'),('frequency.npy','frequency_sha256'),('selection-patterns.json','selection_patterns_sha256')]:check(f'family{family}hash{name}',sha(folder/name)==d[key])
            observability=[]
            for bit,j in enumerate(others):
                pair=candidates.index(tuple(sorted((true,j))));count=int(hit[:,j].sum());strict={str(n):bool(np.any((graph(n,[orbits[true],orbits[j]])>0)&(graph(n,[orbits[true]])==0))) for n in (12,13)}
                check(f'family{family}orbit{j}pair normalizer identity and graph strictness',sizes[pair]==len(population)-count and all(strict.values()))
                observability.append(dict(orbit_index=j,orbit=list(orbits[j]),witnessing_boards=count,observable=count>0,pair_candidate=pair,pair_source_states=sizes[pair],source_relation='strictly_smaller' if count else 'identical',pair_graph_strictly_larger=strict))
            check(f'family{family}all zero-observability cases recorded',observability==d['observability'] and d['all_other_orbits_observable']==all(r['observable'] for r in observability))
            expected=[]
            for mask in range(512):
                observed={others[bit] for bit in range(9) if mask&(1<<bit)};consistent=[cid for cid,chosen in enumerate(candidates) if observed.isdisjoint(chosen)];best=min(sizes[cid] for cid in consistent);winners=[cid for cid in consistent if sizes[cid]==best];recovered=winners==[truecid]
                check(f'family{family}mask{mask}lemma with every tied maximizer',recovered==(mask==511))
                expected.append(dict(mask=mask,consistent_candidates=consistent,maximizers=winners,unique_true_graph=recovered,consistency_envelope_true_graph=mask==511))
            check(f'family{family}all512selection patterns',expected==patterns)
            for dose,outcome in zip(p['doses'],d['outcomes']):
                mass=inclusion_exclusion(freq,dose);total=math.comb(len(population),dose);success=sum(weight for mask,weight in enumerate(mass) if patterns[mask]['unique_true_graph']);gcd=math.gcd(success,total);passed=100*success>=99*total
                check(f'family{family}dose{dose}all512exact joint subset counts',mass==list(map(int,outcome['all512_union_counts'])) and all(v>=0 for v in mass) and sum(mass)==total and success==mass[511])
                check(f'family{family}dose{dose}consistent candidate sampling feasible',all(sizes[cid]>=dose for mask,weight in enumerate(mass) if weight for cid in patterns[mask]['consistent_candidates']))
                row=dict(family=family,dose=dose,primary=dose==16,population_size=len(population),successful_subsets=str(success),total_subsets=str(total),reduced_numerator=str(success//gcd),reduced_denominator=str(total//gcd),unique_true_graph_probability=success/total,unique_true_graph_pct=100*success/total,consistency_envelope_exact_graph_probability=success/total,exact_threshold_pass=passed,decision=('supported' if passed else 'falsified') if dose==16 else 'descriptive')
                check(f'family{family}dose{dose}exact fraction threshold and control identity',row=={k:v for k,v in outcome.items() if k!='all512_union_counts'});rows.append(row)
            check(f'family{family}full distribution receipt',len(d['outcomes'])==5 and d['all_mobius_and_grouped_DP_masses_identical'] and g['source_families'][family]==dict(family=family,distribution_sha256=sha(folder/'exact-distribution.json')))
        primary=[r for r in rows if r['primary']]
        check('all10outcomes and exact twofamily conjunction',rows==g['rows'] and primary==g['primary'] and g['decision']==('supported' if all(r['exact_threshold_pass'] for r in primary) else 'falsified'))
    if require_summary:
        s=read(P/'summary.json')
        check('summary raw identities and exact decimal presentation',s['status']=='validated' and all(s[k]==g[k] for k in ('decision','rows','primary','graph_distinctness')) and s['exact_presentation']==exact_presentation(g['rows']) and s['protocol_sha256']==sha(S/'protocol.json') and s['audit_source_sha256']==sha(S/'analyze.py') and s['verifier_source_sha256']==sha(S/'verify_public.py'))
        for rel,digest in s['public_sha256'].items():check('export '+rel,sha(P/rel)==digest)
        f=read(P/'plot-provenance.json');check('figure input and output provenance',all(sha(R/rel)==digest for rel,digest in f['input_sha256'].items()) and all(sha(P/rel)==digest for rel,digest in f['output_sha256'].items()))
    return dict(status='passed',checks=len(checks),all56_candidate_source_sets_independently_enumerated=True,all_population_witnesses_reconstructed=True,all512_selection_patterns_per_family_and_ties_checked=True,all5120_exact_union_masses_checked_by_direct_inclusion_exclusion=True,all_exact_thresholds_and_control_equivalence_verified=True,elapsed_seconds=time.perf_counter()-start)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[4]);print(json.dumps(verify(parser.parse_args().root)))
