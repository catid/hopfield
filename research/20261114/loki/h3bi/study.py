"""Exact depth-four policy feasibility on every original unresolved checkpoint."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib
import itertools
import resource
import signal
import time
import numpy as np
from common import ROOT,SRC,OUT,frozen,read,save,sha,now,load
from compiler import Budget,compile_tree
from synthetic_gate import run as synthetic_gate


def cpu():
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN);return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime


def main():
    start=cpu();wall=time.perf_counter();p=frozen();OUT.mkdir(parents=True,exist_ok=True);assert not(OUT/'study-result.json').exists();compiled=[];budget=Budget(p['budget']['max_memo_states_total'],p['budget']['max_canonical_split_attempts_total'])
    def expired(*_):raise TimeoutError('H3bi600CPU900wall gate; incomplete not infeasible')
    signal.signal(signal.SIGUSR1,expired)
    try:
        rule=load(SRC.parent/'h3at/source.py','h3bi_rule');orbits=[tuple(o) for o in p['candidate_prior']['orbits']];choices=[x for k in [0,1,2] for x in itertools.combinations(range(55),k)];old=read(ROOT/p['normalizers']['selection_path']);ordinary=np.load(ROOT/p['normalizers']['ordinary_path']);sourcebits=np.load(ROOT/p['normalizers']['bits_path'])['bits'];h12=rule.incidence(ordinary,orbits);B=ROOT/'results/research/loki/h3bc';boards=np.load(B/'ordinary13.npy');bits=np.load(B/'predictions-bits.npy');oldprep=read(B/'preparation.json');census=read(B/'summary.json');prior=read(ROOT/'results/research/loki/h3bf/summary.json');cases=read(ROOT/'results/research/loki/h3bf/case-results.json');h13=rule.incidence(boards,orbits);sourcepred=[];sourcepop={}
        assert census['audit']['all1541N12_masks_and_N13_literal_populations_verified'] and prior['audit']['all1408_source_seeds_and_unique_bank_counts_verified'] and prior['public_sha256']['case-results.json']==sha(ROOT/'results/research/loki/h3bf/case-results.json')
        for cid,chosen in enumerate(choices):
            m12=~h12[:,chosen].any(1) if chosen else np.ones(len(ordinary),bool);columns=ordinary[m12];F12=rule.rule_graph(12,[orbits[j] for j in chosen]);assert np.array_equal(np.packbits(m12,bitorder='little'),sourcebits[cid]) and old['candidates'][cid]==dict(candidate=cid,orbit_indices=list(chosen),orbits=[list(orbits[j]) for j in chosen],complete_source_cardinality=len(columns),source_columns_bytes_sha256=hashlib.sha256(columns.tobytes()).hexdigest(),source_F_bytes_sha256=hashlib.sha256(F12.tobytes()).hexdigest());sourcepred.append(m12);sourcepop[cid]=columns
            m13=~h13[:,chosen].any(1) if chosen else np.ones(len(boards),bool);packed=np.packbits(m13,bitorder='little');F13=rule.rule_graph(13,[orbits[j] for j in chosen]);record=dict(candidate=cid,states=int(m13.sum()),columns_bytes_sha256=hashlib.sha256(boards[m13].tobytes()).hexdigest(),F_bytes_sha256=hashlib.sha256(F13.tobytes()).hexdigest(),packed_bytes_sha256=hashlib.sha256(packed.tobytes()).hexdigest());assert np.array_equal(packed,bits[cid]) and record==oldprep['populations'][cid]
        gate=synthetic_gate(budget);sourcepred=np.array(sourcepred,bool);predictions=np.unpackbits(bits,axis=1,count=len(boards),bitorder='little').astype(bool);lookup={bytes(row):i for i,row in enumerate(ordinary)};groups={};case_map=[]
        for case in cases:
            truth=case['candidate'];indices=np.random.default_rng(case['seed']).choice(len(sourcepop[truth]),16,replace=False);sample=sourcepop[truth][indices];candidates=np.flatnonzero(sourcepred[:,[lookup[bytes(row)] for row in sample]].all(1)).tolist();assert candidates==case['initial_candidates'];queried=[]
            for step in case['steps']:
                index=step['query_index'];label=bool(predictions[truth,index]);remaining=[cid for cid in candidates if bool(predictions[cid,index])==label];assert index not in queried and step['label']==label and step['candidates_before']==candidates and step['candidates_after']==remaining;candidates=remaining;queried.append(index)
            assert candidates==case['remaining_candidates'] and case['exact_true_singleton']==(candidates==[truth]);key=tuple(candidates)
            if not case['exact_true_singleton']:
                assert 2<=len(candidates)<=16 and len(queried)==4 and all(len(set(predictions[candidates,index].tolist()))==1 for index in queried);groups.setdefault(key,[]).append(case['case'])
            case_map.append(dict(case=case['case'],known_four_query_success=case['exact_true_singleton'],remaining_candidates=candidates,original_queried_board_indices=queried,tree_group=None))
        entries=[dict(group=i,candidates=list(key),case_ids=groups[key]) for i,key in enumerate(sorted(groups))]
        for entry in entries:
            for case in entry['case_ids']:case_map[case]['tree_group']=entry['group']
        assert [r['case'] for r in case_map if not r['known_four_query_success']]==p['primary_checkpoint_case_ids'];save(OUT/'checkpoint-groups.json',dict(utc=now(),groups=entries,case_map=case_map,all1408_cases_retained=True,all704_unresolved_mapped=True,synthetic_gate_sha256=sha(OUT/'synthetic-gate.json'),protocol_sha256=sha(SRC/'protocol.json')));print('CHECKPOINT_GROUPS',len(entries),'cases704',flush=True)
        for entry in entries:
            candidates=entry['candidates'];folder=OUT/f'group{entry["group"]}';folder.mkdir();catalog_start=cpu();catalog_wall=time.perf_counter();codes=np.zeros(len(boards),np.uint32)
            for local,cid in enumerate(candidates):codes|=predictions[cid].astype(np.uint32)<<local
            signatures,first,counts=np.unique(codes,return_index=True,return_counts=True);catalog=[dict(signature=int(code),lexfirst_board_index=int(index),board_count=int(count)) for code,index,count in zip(signatures,first,counts)];save(folder/'catalog.json',dict(candidates=candidates,universe_boards=len(boards),signatures=catalog));save(folder/'compiler-input.json',dict(candidates=candidates,catalog_sha256=sha(folder/'catalog.json'),depth=4,allowed_inputs_only=True));before_states=budget.states;before_splits=budget.splits;compile_start=cpu();compile_wall=time.perf_counter();proof=compile_tree(candidates,catalog,4,budget);compile_cpu=cpu()-compile_start;compile_elapsed=time.perf_counter()-compile_wall;save(folder/'proof.json',proof)
            if proof['feasible']:
                all_queries=set()
                for local,cid in enumerate(candidates):
                    key=proof['root_state'];used=[]
                    while proof['states'][key]['reason']!='singleton_leaf':
                        node=proof['states'][key];index=node['chosen_split']['board_index'];assert index not in used;used.append(index);all_queries.add(index);key=node['positive_state'] if predictions[cid,index] else node['negative_state']
                    assert proof['states'][key]['mask']==1<<local and len(used)==proof['leaf_depths'][str(cid)]<=4
                assert all(not all_queries.intersection(case_map[case]['original_queried_board_indices']) for case in entry['case_ids'])
            row=dict(entry,feasible=proof['feasible'],worst_leaf_depth=proof['worst_leaf_depth'],signature_count=len(catalog),memo_states=budget.states-before_states,canonical_split_attempts=budget.splits-before_splits,catalog_and_compilation_CPU_seconds=cpu()-catalog_start,catalog_and_compilation_wall_seconds=time.perf_counter()-catalog_wall,compilation_CPU_seconds=compile_cpu,compilation_wall_seconds=compile_elapsed,catalog_sha256=sha(folder/'catalog.json'),compiler_input_sha256=sha(folder/'compiler-input.json'),proof_sha256=sha(folder/'proof.json'));compiled.append(row);save(folder/'result.json',row);print('TREE',entry['group'],'K',len(candidates),'signatures',len(catalog),'feasible',proof['feasible'],'states',row['memo_states'],'splits',row['canonical_split_attempts'],flush=True)
        feasible_cases=sum(len(r['case_ids']) for r in compiled if r['feasible']);decision='supported' if feasible_cases==704 else 'falsified';save(OUT/'study-result.json',dict(status='complete_verified',utc=now(),decision=decision,groups=compiled,distinct_checkpoint_sets=len(entries),unresolved_cases=704,feasible_checkpoint_cases=feasible_cases,infeasible_checkpoint_cases=704-feasible_cases,known_success_cases=704,global_memo_states=budget.states,global_canonical_split_attempts=budget.splits,new_source_draws=0,new_oracle_labels=0,checkpoint_groups_sha256=sha(OUT/'checkpoint-groups.json'),synthetic_gate_sha256=sha(OUT/'synthetic-gate.json'),stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
    except BaseException as exc:
        save(OUT/'study-result.json',dict(status='failed_or_incomplete',utc=now(),error=repr(exc),completed_groups=compiled,global_memo_states=budget.states,global_canonical_split_attempts=budget.splits,partial_results_usable=False,cap_is_not_infeasibility=True,stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json')));raise


if __name__=='__main__':main()
