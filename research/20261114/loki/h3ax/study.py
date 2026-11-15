"""Fresh positive-only identification on a fixed grid of genuine two-orbit truths."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib
import itertools
import math
import resource
import signal
import time
import numpy as np
from common import ROOT,SRC,OUT,frozen,read,save,sha,now,load


def cpu():
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN);return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime


def main():
    start=cpu();wall=time.perf_counter();p=frozen();OUT.mkdir(parents=True,exist_ok=True);assert not(OUT/'study-result.json').exists();rows=[];families=[]
    def expired(*_):raise TimeoutError('H3ax120CPU180wall gate')
    signal.signal(signal.SIGUSR1,expired)
    try:
        helper=load(SRC.parent/'h3at/source.py','h3ax_rules');old=read(ROOT/p['normalizers']['selection_path']);ordinary=np.load(ROOT/p['normalizers']['ordinary_path']);bits=np.load(ROOT/p['normalizers']['bits_path'])['bits'];orbits=[tuple(o) for o in p['candidate_prior']['orbits']];choices=[chosen for k in (0,1,2) for chosen in itertools.combinations(range(55),k)];hit=helper.incidence(ordinary,orbits);candidates=[];graphs=[];normalizers=[];populations=[]
        for cid,chosen in enumerate(choices):
            mask=~hit[:,chosen].any(1) if chosen else np.ones(len(ordinary),bool);columns=ordinary[mask];F=helper.rule_graph(12,[orbits[j] for j in chosen]);c=old['candidates'][cid];assert np.array_equal(np.packbits(mask,bitorder='little'),bits[cid]);assert c==dict(candidate=cid,orbit_indices=list(chosen),orbits=[list(orbits[j]) for j in chosen],complete_source_cardinality=len(columns),source_columns_bytes_sha256=hashlib.sha256(columns.tobytes()).hexdigest(),source_F_bytes_sha256=hashlib.sha256(F.tobytes()).hexdigest());M=len(columns);denom=math.comb(M,16) if M>=16 else None;normalizers.append(denom);populations.append(columns);graphs.append(helper.rule_graph(13,[orbits[j] for j in chosen]));candidates.append(dict(c,orbit_count=len(chosen),models_in_class=math.comb(55,len(chosen)),likelihood_denominator_if_eligible=str(denom) if denom is not None else None,MAP_integer_score_without_common3=str(math.comb(55,len(chosen))*denom) if denom is not None else None))
        assert len({F.tobytes() for F in graphs})==1541
        save(OUT/'candidate-normalizers.json',dict(candidates=candidates,all1541_complete_source_bitsets_cardinalities_column_and_graph_hashes_rechecked=True,all1541_N13_graphs_distinct=True,independent_literal_normalizer_verification_reused_from='H3at',protocol_sha256=sha(SRC/'protocol.json')))
        for cfg in p['families']:
            family=cfg['index'];true_indices=tuple(sorted(orbits.index(tuple(o)) for o in cfg['true_orbits']));truecid=choices.index(true_indices);population=populations[truecid];M=len(population);folder=OUT/f'f{family}';folder.mkdir(exist_ok=True);np.save(folder/'source-population.npy',population);np.save(folder/'true-N13-F.npy',graphs[truecid]);eligible=M>=16;familyrows=[]
            if eligible:
                for bank,seed in enumerate(cfg['seeds']):
                    indices=np.random.default_rng(seed).choice(M,16,replace=False);columns=population[indices];seen=helper.incidence(columns,orbits);scores=[]
                    for cid,chosen in enumerate(choices):
                        violations=int(seen[:,chosen].any(1).sum()) if chosen else 0;consistent=violations==0;assert not consistent or len(populations[cid])>=16;scores.append(dict(candidate=cid,sample_violations=violations,consistent=consistent,likelihood_denominator=str(normalizers[cid]) if consistent else None,MAP_integer_score_without_common3=candidates[cid]['MAP_integer_score_without_common3'] if consistent else None))
                    selection={}
                    for arm,key in [('ML','likelihood_denominator'),('MAP','MAP_integer_score_without_common3')]:
                        best=min(int(r[key]) for r in scores if r['consistent']);winners=[r['candidate'] for r in scores if r['consistent'] and int(r[key])==best];exact=all(np.array_equal(graphs[cid],graphs[truecid]) for cid in winners);selection[arm]=dict(selected_candidates=winners,best_integer_score=str(best),unique=len(winners)==1,every_tie_exact_N13_graph=exact,selected_orbit_counts=[len(choices[cid]) for cid in winners],contains_true_candidate=truecid in winners)
                    out=folder/f'b{bank}';out.mkdir();np.savez_compressed(out/'draw.npz',indices=indices,columns=columns,orbit_incidence=seen);save(out/'scores.json',dict(family=family,bank=bank,seed=seed,scores=scores,selections=selection));row=dict(family=family,bank=bank,seed=seed,population_size=M,true_candidate=truecid,ML=selection['ML'],MAP=selection['MAP'],draw_sha256=sha(out/'draw.npz'),scores_sha256=sha(out/'scores.json'));rows.append(row);familyrows.append(row)
                ML=sum(r['ML']['every_tie_exact_N13_graph'] for r in familyrows);MAP=sum(r['MAP']['every_tie_exact_N13_graph'] for r in familyrows);decision='supported' if MAP>=3 and MAP>=ML else 'falsified'
            else:ML=MAP=None;decision='ineligible_source_fewer_than16'
            result=dict(family=family,true_orbits=cfg['true_orbits'],true_candidate=truecid,source_population_size=M,eligible=eligible,drawn_banks=len(familyrows),ML_exact_banks=ML,MAP_exact_banks=MAP,required_MAP_exact_banks=3,requires_MAP_at_least_ML=True,decision=decision,source_population_sha256=sha(folder/'source-population.npy'),true_N13_F_sha256=sha(folder/'true-N13-F.npy'));families.append(result);print('FAMILY',family,cfg['true_orbits'],M,ML,MAP,decision,flush=True)
        eligible=[f for f in families if f['eligible']];overall='inconclusive_no_eligible_families' if not eligible else 'supported' if all(f['decision']=='supported' for f in eligible) else 'falsified'
        save(OUT/'study-result.json',dict(status='complete_verified',utc=now(),decision=overall,family_results=families,rows=rows,fixed_grid_families=45,eligible_families=len(eligible),ineligible_families=45-len(eligible),new_positive_draws=len(rows),candidate_normalizers_sha256=sha(OUT/'candidate-normalizers.json'),all_ties_retained=True,no_target_capacity_or_state_enumeration=True,stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
    except BaseException as exc:
        save(OUT/'study-result.json',dict(status='failed_or_incomplete',utc=now(),error=repr(exc),completed_rows=rows,completed_families=families,partial_results_usable=False,stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json')));raise


if __name__=='__main__':main()
