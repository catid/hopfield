"""Exact N13 state-prediction ceilings, not an acquisition-policy experiment."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib
import itertools
import resource
import signal
import time
import numpy as np
from common import ROOT,SRC,OUT,frozen,read,save,sha,now,load


def cpu():
    a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN);return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime


def main():
    start=cpu();wall=time.perf_counter();p=frozen();OUT.mkdir(parents=True,exist_ok=True);assert not(OUT/'study-result.json').exists();results=[];pairs=[]
    def expired(*_):raise TimeoutError('H3bg120CPU180wall gate')
    signal.signal(signal.SIGUSR1,expired)
    try:
        rule=load(SRC.parent/'h3at/source.py','h3bg_rule');orbits=[tuple(o) for o in p['candidate_prior']['orbits']];choices=[x for k in [0,1,2] for x in itertools.combinations(range(55),k)];old=read(ROOT/p['normalizers']['selection_path']);ordinary=np.load(ROOT/p['normalizers']['ordinary_path']);sourcebits=np.load(ROOT/p['normalizers']['bits_path'])['bits'];hit12=rule.incidence(ordinary,orbits);B=ROOT/'results/research/loki/h3bc';boards=np.load(B/'ordinary13.npy');bits=np.load(B/'predictions-bits.npy');oldprep=read(B/'preparation.json');census=read(B/'summary.json');prior=read(ROOT/'results/research/loki/h3bf/summary.json');cases=read(ROOT/'results/research/loki/h3bf/case-results.json');hit13=rule.incidence(boards,orbits);sourcepred=[];sourcepop={};classes=[];class_index={};model_class={};population_records=[]
        assert census['audit']['all1541N12_masks_and_N13_literal_populations_verified'] and prior['audit']['all1408_source_seeds_and_unique_bank_counts_verified'] and prior['public_sha256']['case-results.json']==sha(ROOT/'results/research/loki/h3bf/case-results.json')
        for cid,chosen in enumerate(choices):
            m12=~hit12[:,chosen].any(1) if chosen else np.ones(len(ordinary),bool);columns=ordinary[m12];F12=rule.rule_graph(12,[orbits[j] for j in chosen]);assert np.array_equal(np.packbits(m12,bitorder='little'),sourcebits[cid]) and old['candidates'][cid]==dict(candidate=cid,orbit_indices=list(chosen),orbits=[list(orbits[j]) for j in chosen],complete_source_cardinality=len(columns),source_columns_bytes_sha256=hashlib.sha256(columns.tobytes()).hexdigest(),source_F_bytes_sha256=hashlib.sha256(F12.tobytes()).hexdigest());sourcepred.append(m12);sourcepop[cid]=columns
            m13=~hit13[:,chosen].any(1) if chosen else np.ones(len(boards),bool);F13=rule.rule_graph(13,[orbits[j] for j in chosen]);packed=np.packbits(m13,bitorder='little');record=dict(candidate=cid,states=int(m13.sum()),columns_bytes_sha256=hashlib.sha256(boards[m13].tobytes()).hexdigest(),F_bytes_sha256=hashlib.sha256(F13.tobytes()).hexdigest(),packed_bytes_sha256=hashlib.sha256(packed.tobytes()).hexdigest());assert np.array_equal(packed,bits[cid]) and record==oldprep['populations'][cid];population_records.append(record);key=packed.tobytes()
            if key not in class_index:class_index[key]=len(classes);classes.append(dict(class_id=len(classes),states=record['states'],prediction_bits_sha256=record['packed_bytes_sha256'],members=[]))
            group=class_index[key];classes[group]['members'].append(cid);model_class[cid]=group
        partition=dict(utc=now(),classes=classes,candidate_to_class=[model_class[cid] for cid in range(1541)],all1541models_retained=True,empty_models=[r['candidate'] for r in population_records if r['states']==0],protocol_sha256=sha(SRC/'protocol.json'));save(OUT/'N13-partition.json',partition);print('N13_PARTITION',len(classes),'classes',sum(len(c['members'])>1 for c in classes),'nontrivial',flush=True);sourcepred=np.array(sourcepred,bool);predictions=np.unpackbits(bits,axis=1,count=len(boards),bitorder='little').astype(bool);lookup={bytes(row):i for i,row in enumerate(ordinary)};needed=set()
        for case in cases:
            truth=case['candidate'];indices=np.random.default_rng(case['seed']).choice(len(sourcepop[truth]),16,replace=False);sample=sourcepop[truth][indices];initial=np.flatnonzero(sourcepred[:,[lookup[bytes(row)] for row in sample]].all(1)).tolist();key=hashlib.sha256(b''.join(sorted(map(bytes,sample)))).hexdigest();assert initial==case['initial_candidates'] and key==case['unordered_sample_sha256'];aliases=[cid for cid in initial if model_class[cid]==model_class[truth]];identifiable=aliases==[truth];known=case['exact_true_singleton'];assert not known or identifiable
            needed.update((truth,cid) for cid in initial if cid!=truth);results.append(dict(case=case['case'],candidate=truth,bank=case['bank'],seed=case['seed'],initial_candidates=initial,observed_four_query_exact=known,diagnostic_scope='known_success_consistency' if known else 'new_failed_case_ceiling',global_N13_class=model_class[truth],N13_indistinguishable_initial_candidates=aliases,N13_identifiable_in_principle=identifiable,source_unordered_sample_sha256=key))
        assert len(needed)<=p['budget']['max_distinct_truth_competitor_pairs']
        for truth,other in sorted(needed):
            directions=[]
            for valid,invalid in [(truth,other),(other,truth)]:
                indices=np.flatnonzero(predictions[valid]&~predictions[invalid]);first=int(indices[0]) if len(indices) else None;directions.append(dict(valid_candidate=valid,invalid_candidate=invalid,states=len(indices),lexfirst_board_index=first,lexfirst_columns_zero_based=boards[first].tolist() if first is not None else None))
            pair=dict(true_candidate=truth,other_initial_candidate=other,predictions_identical=model_class[truth]==model_class[other],directions=directions);assert pair['predictions_identical']==all(d['states']==0 for d in directions);pairs.append(pair)
        save(OUT/'pair-difference-witnesses.json',dict(utc=now(),pairs=pairs,all_required_pairs_retained=True,truth_aware_diagnostic_not_query_policy=True));save(OUT/'case-ceilings.json',results);truth_results=[]
        for cfg in p['truths']:
            rows=[r for r in results if r['candidate']==cfg['candidate']];failed=[r for r in rows if not r['observed_four_query_exact']];truth_results.append(dict(candidate=cfg['candidate'],orbits=cfg['orbits'],cases=len(rows),observed_four_query_exact=sum(r['observed_four_query_exact'] for r in rows),known_success_identifiable=sum(r['observed_four_query_exact'] and r['N13_identifiable_in_principle'] for r in rows),failed_cases=len(failed),failed_cases_identifiable=sum(r['N13_identifiable_in_principle'] for r in failed),all_case_identifiability_ceiling=sum(r['N13_identifiable_in_principle'] for r in rows),unavoidable_ambiguous_cases=sum(not r['N13_identifiable_in_principle'] for r in rows)))
        failed=[r for r in results if not r['observed_four_query_exact']];assert [r['case'] for r in failed]==p['primary_failed_case_ids'];decision='supported' if all(r['N13_identifiable_in_principle'] for r in failed) else 'falsified';save(OUT/'study-result.json',dict(status='complete_verified',utc=now(),decision=decision,truth_results=truth_results,cases=len(results),known_success_cases=len(results)-len(failed),failed_cases=len(failed),failed_cases_identifiable=sum(r['N13_identifiable_in_principle'] for r in failed),unavoidable_ambiguous_cases=sum(not r['N13_identifiable_in_principle'] for r in results),global_N13_equivalence_classes=len(classes),global_nontrivial_classes=sum(len(c['members'])>1 for c in classes),distinct_truth_competitor_pairs=len(pairs),new_oracle_labels=0,partition_sha256=sha(OUT/'N13-partition.json'),cases_sha256=sha(OUT/'case-ceilings.json'),pair_witnesses_sha256=sha(OUT/'pair-difference-witnesses.json'),stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json'),source_sha256=sha(__file__)))
    except BaseException as exc:
        save(OUT/'study-result.json',dict(status='failed_or_incomplete',utc=now(),error=repr(exc),completed_cases=results,completed_pairs=pairs,partial_results_usable=False,stage_cpu_seconds=cpu()-start,stage_wall_seconds=time.perf_counter()-wall,protocol_sha256=sha(SRC/'protocol.json')));raise


if __name__=='__main__':main()
