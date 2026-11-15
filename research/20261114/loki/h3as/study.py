"""Finite-population probability of unique true-rule selection under the H3ar prior."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import itertools
import math
import resource
import signal
import time
import numpy as np
from common import ROOT,SRC,OUT,frozen,read,save,sha,now
from distribution import mobius,grouped_dp,validate


def graph(n,orbits):
    cells=np.arange(n*n);dr=abs(cells[:,None]//n-cells[None,:]//n);dc=abs(cells[:,None]%n-cells[None,:]%n)
    F=(dr==0)|(dc==0)|(dr==dc)
    for a,b in orbits: F|=((dr==a)&(dc==b))|((dr==b)&(dc==a))
    np.fill_diagonal(F,False);return F


def incidence(columns,orbits):
    a,b=np.triu_indices(columns.shape[1],1);dr=b-a;dc=abs(columns[:,b].astype(np.int16)-columns[:,a].astype(np.int16))
    return np.stack([(((dr==i)&(dc==j))|((dr==j)&(dc==i))).any(1) for i,j in orbits],axis=1)


def cpu():
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime


def main():
    start=cpu();wall=time.perf_counter();p=frozen();OUT.mkdir(parents=True,exist_ok=True);assert not(OUT/'study-result.json').exists();rows=[]
    def expired(*_): raise TimeoutError('H3as 120 CPU / 180 wall budget')
    signal.signal(signal.SIGUSR1,expired)
    try:
        validation=validate();orbits=[tuple(o) for o in p['candidate_prior']['orbits']];candidates=[chosen for size in (0,1,2) for chosen in itertools.combinations(range(10),size)]
        inherited=read(ROOT/p['source_selection_path']);ordinary=np.load(ROOT/p['ordinary_source_path']);hit=incidence(ordinary,orbits)
        assert len(ordinary)==14200 and len(candidates)==56
        sizes=[];graph_checks={}
        for n in (12,13):
            signatures=[graph(n,[orbits[j] for j in chosen]).tobytes() for chosen in candidates]
            graph_checks[str(n)]=dict(distinct_graphs=len(set(signatures)),candidate_graphs=len(signatures))
            assert len(set(signatures))==56,'Candidate graphs not distinct; frozen lemma gate fails'
        for cid,chosen in enumerate(candidates):
            allowed=ordinary[~hit[:,chosen].any(1)] if chosen else ordinary
            assert np.array_equal(allowed,np.load(ROOT/f'results/research/loki/h3ar/candidate{cid}/source-columns.npy'))
            assert len(allowed)==inherited['candidates'][cid]['complete_source_cardinality'];sizes.append(len(allowed))
        for family,cfg in enumerate(p['families']):
            population=np.load(ROOT/cfg['source_path']);true=orbits.index(tuple(cfg['leaper']));others=[j for j in range(10) if j!=true];truecid=candidates.index((true,));hits=incidence(population,orbits)
            assert np.array_equal(population,ordinary[~hit[:,true]]) and not hits[:,true].any() and len(population)==sizes[truecid]<sizes[0]
            masks=(hits[:,others].astype(np.int64)*(1<<np.arange(9))).sum(1);freq=np.bincount(masks,minlength=512);folder=OUT/f'family{family}';folder.mkdir(exist_ok=True)
            np.save(folder/'population.npy',population);np.save(folder/'witness-masks.npy',masks);np.save(folder/'frequency.npy',freq)
            observability=[]
            for bit,j in enumerate(others):
                pair=candidates.index(tuple(sorted((true,j))));count=int(hits[:,j].sum());assert sizes[pair]==len(population)-count
                strict={str(n):bool(np.any(graph(n,[orbits[true],orbits[j]])&~graph(n,[orbits[true]]))) for n in (12,13)};assert all(strict.values())
                observability.append(dict(orbit_index=j,orbit=list(orbits[j]),witnessing_boards=count,observable=count>0,pair_candidate=pair,pair_source_states=sizes[pair],source_relation='strictly_smaller' if count else 'identical',pair_graph_strictly_larger=strict))
            patterns=[]
            for mask in range(512):
                observed={others[bit] for bit in range(9) if mask&(1<<bit)};consistent=[cid for cid,chosen in enumerate(candidates) if not observed.intersection(chosen)];best=min(sizes[cid] for cid in consistent);winners=[cid for cid in consistent if sizes[cid]==best]
                recovered=winners==[truecid];assert recovered==(mask==511)
                patterns.append(dict(mask=mask,consistent_candidates=consistent,maximizers=winners,unique_true_graph=recovered,consistency_envelope_true_graph=mask==511))
            save(folder/'selection-patterns.json',patterns)
            dp,operations=grouped_dp(freq.tolist(),max(p['doses']),progress=True);outcomes=[]
            for dose in p['doses']:
                mass=mobius(freq.tolist(),dose);direct=[dp[dose].get(mask,0) for mask in range(512)];assert mass==direct
                total=math.comb(len(population),dose);success=sum(value for mask,value in enumerate(mass) if patterns[mask]['unique_true_graph']);assert success==mass[511]
                for mask,value in enumerate(mass):
                    if value: assert all(sizes[cid]>=dose for cid in patterns[mask]['consistent_candidates'])
                gcd=math.gcd(success,total);passed=100*success>=99*total
                outcome=dict(family=family,dose=dose,primary=dose==16,population_size=len(population),successful_subsets=str(success),total_subsets=str(total),reduced_numerator=str(success//gcd),reduced_denominator=str(total//gcd),unique_true_graph_probability=success/total,unique_true_graph_pct=100*success/total,consistency_envelope_exact_graph_probability=success/total,exact_threshold_pass=passed,decision=('supported' if passed else 'falsified') if dose==16 else 'descriptive',all512_union_counts=[str(value) for value in mass]);outcomes.append(outcome);rows.append({k:v for k,v in outcome.items() if k!='all512_union_counts'})
                print('EXACT_RELIABILITY',family,dose,outcome['unique_true_graph_pct'],outcome['decision'],flush=True)
            save(folder/'exact-distribution.json',dict(family=family,population_size=len(population),true_orbit_index=true,other_orbit_indices=others,true_candidate=truecid,all_other_orbits_observable=all(r['observable'] for r in observability),observability=observability,outcomes=outcomes,independent_grouped_DP_operations=operations,all_mobius_and_grouped_DP_masses_identical=True,population_sha256=sha(folder/'population.npy'),witness_masks_sha256=sha(folder/'witness-masks.npy'),frequency_sha256=sha(folder/'frequency.npy'),selection_patterns_sha256=sha(folder/'selection-patterns.json')))
        primary=[r for r in rows if r['primary']]
        save(OUT/'study-result.json',dict(status='complete_verified',utc=now(),decision='supported' if all(r['exact_threshold_pass'] for r in primary) else 'falsified',rows=rows,primary=primary,validation=validation,graph_distinctness=graph_checks,all56_source_cardinalities_verified=True,all512_abstract_pattern_lemma_checks_per_family=True,all_zero_observability_cases_retained=True,source_families=[dict(family=i,distribution_sha256=sha(OUT/f'family{i}/exact-distribution.json')) for i in range(2)],stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
    except BaseException as exc:
        save(OUT/'study-result.json',dict(status='failed_or_incomplete',utc=now(),error=repr(exc),completed_rows=rows,partial_results_usable=False,stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json')));raise


if __name__=='__main__':main()
