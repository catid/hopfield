"""Exact expected-cost optimization on every examined H3bj checkpoint group."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from fractions import Fraction
import hashlib
import itertools
import math
import resource
import signal
import time
import numpy as np
from common import ROOT,SRC,OUT,frozen,read,save,sha,now,load
from compiler import Budget,compile_tree
from weights import integer_weights
from synthetic_gate import run as synthetic_gate


def cpu():
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN);return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime


def fraction(x):return dict(numerator=str(x.numerator),denominator=str(x.denominator),display_float=float(x))


def main():
    start=cpu();wall=time.perf_counter();p=frozen();OUT.mkdir(parents=True,exist_ok=True);assert not(OUT/'study-result.json').exists();compiled=[];budget=Budget(p['budget']['max_memo_states_total'],p['budget']['max_canonical_split_attempts_total'])
    def expired(*_):raise TimeoutError('H3bl600CPU900wall; whole-study incomplete')
    signal.signal(signal.SIGUSR1,expired)
    try:
        rule=load(SRC.parent/'h3at/source.py','h3bl_rule');orbits=[tuple(o) for o in p['candidate_prior']['orbits']];choices=[x for k in [0,1,2] for x in itertools.combinations(range(55),k)];old=read(ROOT/p['normalizers']['selection_path']);ordinary=np.load(ROOT/p['normalizers']['ordinary_path']);sourcebits=np.load(ROOT/p['normalizers']['bits_path'])['bits'];h12=rule.incidence(ordinary,orbits);B=ROOT/'results/research/loki/h3bc';J=ROOT/'results/research/loki/h3bj';boards=np.load(B/'ordinary13.npy');bits=np.load(B/'predictions-bits.npy');oldprep=read(B/'preparation.json');census=read(B/'summary.json');prior=read(J/'summary.json');cases=read(J/'case-results.json');groups=read(J/'groups.json');h13=rule.incidence(boards,orbits);scores=[]
        assert census['audit']['all1541N12_masks_and_N13_literal_populations_verified'] and prior['audit']['all1408_fresh_source_seeds_and_duplicate_checks_replayed'] and prior['audit']['all_groups_complete_catalogs_and_certificates_verified']==125
        for cid,chosen in enumerate(choices):
            m12=~h12[:,chosen].any(1) if chosen else np.ones(len(ordinary),bool);columns=ordinary[m12];F12=rule.rule_graph(12,[orbits[j] for j in chosen]);assert np.array_equal(np.packbits(m12,bitorder='little'),sourcebits[cid]) and old['candidates'][cid]==dict(candidate=cid,orbit_indices=list(chosen),orbits=[list(orbits[j]) for j in chosen],complete_source_cardinality=len(columns),source_columns_bytes_sha256=hashlib.sha256(columns.tobytes()).hexdigest(),source_F_bytes_sha256=hashlib.sha256(F12.tobytes()).hexdigest());scores.append(str(math.comb(55,len(chosen))*math.comb(len(columns),16)) if len(columns)>=16 else None)
            m13=~h13[:,chosen].any(1) if chosen else np.ones(len(boards),bool);packed=np.packbits(m13,bitorder='little');F13=rule.rule_graph(13,[orbits[j] for j in chosen]);record=dict(candidate=cid,states=int(m13.sum()),columns_bytes_sha256=hashlib.sha256(boards[m13].tobytes()).hexdigest(),F_bytes_sha256=hashlib.sha256(F13.tobytes()).hexdigest(),packed_bytes_sha256=hashlib.sha256(packed.tobytes()).hexdigest());assert np.array_equal(packed,bits[cid]) and record==oldprep['populations'][cid]
        assert scores==read(J/'score-values.json');synthetic_gate(budget);predictions=np.unpackbits(bits,axis=1,count=len(boards),bitorder='little').astype(bool);mapping=[dict(case=r['case'],original_tree_group=r['tree_group'],checkpoint_candidates=r['checkpoint_candidates'],known_four_query_success=len(r['checkpoint_candidates'])==1) for r in cases];assert len(mapping)==1408 and sum(not r['known_four_query_success'] for r in mapping)==701;save(OUT/'checkpoint-mappings.json',dict(utc=now(),cases=mapping,all1408_retained=True,unresolved701_retained=True,protocol_sha256=sha(SRC/'protocol.json')));aggregate_old=Fraction(0);aggregate_new=Fraction(0)
        for original in groups:
            group=original['group'];candidates=original['candidates'];folder=OUT/f'group{group}';folder.mkdir();before_states=budget.states;before_splits=budget.splits;begin=cpu();beginwall=time.perf_counter();weights,weight_receipt=integer_weights(candidates,scores,p['budget']['max_LCM_or_weight_bits']);save(folder/'weights.json',weight_receipt);codes=np.zeros(len(boards),np.uint32)
            for local,cid in enumerate(candidates):codes|=predictions[cid].astype(np.uint32)<<local
            signatures,first,counts=np.unique(codes,return_index=True,return_counts=True);catalog=[dict(signature=int(code),lexfirst_board_index=int(index),board_count=int(count)) for code,index,count in zip(signatures,first,counts)];stored=dict(candidates=candidates,universe_boards=len(boards),signatures=catalog);assert stored==read(J/f'group{group}/catalog.json');save(folder/'catalog.json',stored);save(folder/'compiler-input.json',dict(candidates=candidates,weights=[str(w) for w in weights],catalog_sha256=sha(folder/'catalog.json'),depth=4,allowed_inputs_only=True));proof=compile_tree(candidates,weights,catalog,4,budget);assert proof['feasible'];save(folder/'proof.json',proof);oldproof=read(J/f'group{group}/proof.json');oldcost=sum(w*oldproof['leaf_depths'][str(cid)] for cid,w in zip(candidates,weights));newcost=int(proof['optimal_cost']);assert newcost<=oldcost
            for local,cid in enumerate(candidates):
                state=proof['root_state'];used=[]
                while proof['states'][state]['reason']!='singleton_leaf':
                    node=proof['states'][state];index=node['chosen_split']['board_index'];assert index not in used;used.append(index);state=node['positive_state'] if predictions[cid,index] else node['negative_state']
                assert proof['states'][state]['mask']==1<<local and len(used)==proof['leaf_depths'][str(cid)]<=4
            case_ids=[r['case'] for r in cases if r['tree_group']==group];old_expected=Fraction(oldcost,sum(weights));new_expected=Fraction(newcost,sum(weights));aggregate_old+=len(case_ids)*old_expected;aggregate_new+=len(case_ids)*new_expected;row=dict(group=group,candidates=candidates,case_ids=case_ids,case_multiplicity=len(case_ids),old_weighted_depth_cost=str(oldcost),optimal_weighted_depth_cost=str(newcost),weight_sum=str(sum(weights)),old_expected_additional_queries=fraction(old_expected),optimal_expected_additional_queries=fraction(new_expected),strict_expected_improvement=newcost<oldcost,old_worst_leaf_depth=oldproof['worst_leaf_depth'],optimal_worst_leaf_depth=proof['worst_leaf_depth'],signature_count=len(catalog),memo_states=budget.states-before_states,canonical_split_attempts=budget.splits-before_splits,CPU_seconds=cpu()-begin,wall_seconds=time.perf_counter()-beginwall,weights_sha256=sha(folder/'weights.json'),catalog_sha256=sha(folder/'catalog.json'),compiler_input_sha256=sha(folder/'compiler-input.json'),proof_sha256=sha(folder/'proof.json'));save(folder/'result.json',row);compiled.append(row);print('OPTIMAL',group,'K',len(candidates),'states',row['memo_states'],'splits',row['canonical_split_attempts'],'strict',row['strict_expected_improvement'],flush=True)
        descriptive=[]
        for case in cases:
            group=case['tree_group'];truth=case['candidate'];new=0 if group is None else int(read(OUT/f'group{group}/proof.json')['leaf_depths'][str(truth)]);old=case['hybrid_additional_calls'];descriptive.append(dict(case=case['case'],candidate=truth,tree_group=group,shared_calls=case['shared_membership_calls'],old_tree_continuation_calls=old,optimized_tree_realized_truth_depth=new,baseline_MAP_continuation_calls=case['baseline_additional_calls'],old_hybrid_policy_cost=case['hybrid_policy_label_cost'],optimized_hybrid_realized_truth_cost=case['shared_membership_calls']+new,baseline_MAP_policy_cost=case['baseline_policy_label_cost'],retrospective_no_new_label_acquisition=True))
        save(OUT/'retrospective-truth-costs.json',descriptive);decision='supported' if 10*aggregate_new<=9*aggregate_old else 'falsified';reduction=1-aggregate_new/aggregate_old;save(OUT/'study-result.json',dict(status='complete_verified',utc=now(),decision=decision,groups=compiled,distinct_checkpoint_sets=125,unresolved_cases=701,known_four_query_success_cases=707,case_weighted_expected_old_sum=fraction(aggregate_old),case_weighted_expected_optimal_sum=fraction(aggregate_new),mean_expected_old_continuation=fraction(aggregate_old/701),mean_expected_optimal_continuation=fraction(aggregate_new/701),relative_expected_reduction=fraction(reduction),exact_threshold_comparison=dict(left_10_times_optimal=fraction(10*aggregate_new),right_9_times_old=fraction(9*aggregate_old),passed=10*aggregate_new<=9*aggregate_old),strictly_improved_sets=sum(r['strict_expected_improvement'] for r in compiled),strictly_improved_checkpoint_cases=sum(r['case_multiplicity'] for r in compiled if r['strict_expected_improvement']),guaranteed_nonincrease_is_correctness_not_hypothesis=True,descriptive_old_continuation_cost=sum(r['old_tree_continuation_calls'] for r in descriptive),descriptive_optimized_continuation_cost=sum(r['optimized_tree_realized_truth_depth'] for r in descriptive),descriptive_baseline_MAP_continuation_cost=sum(r['baseline_MAP_continuation_calls'] for r in descriptive),global_memo_states=budget.states,global_canonical_split_attempts=budget.splits,new_source_draws=0,new_oracle_labels=0,checkpoint_mappings_sha256=sha(OUT/'checkpoint-mappings.json'),retrospective_truth_costs_sha256=sha(OUT/'retrospective-truth-costs.json'),synthetic_gate_sha256=sha(OUT/'synthetic-gate.json'),stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
    except BaseException as exc:
        save(OUT/'study-result.json',dict(status='failed_or_incomplete',utc=now(),error=repr(exc),completed_groups=compiled,global_memo_states=budget.states,global_canonical_split_attempts=budget.splits,partial_results_usable=False,cap_is_not_infeasibility=True,stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json')));raise


if __name__=='__main__':main()
