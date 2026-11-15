"""Oracle-only randomized full truth schedule and eight-positive source banks."""
import hashlib
import numpy as np
from common import save,sha


def draw(p,ordinary,sourcepred,sourcepop,score_tables,history,ledger,folder,statuses,sources=None):
    eligible=[r['candidate'] for r in p['truths'] if r['eligible']]
    assert len(eligible)==1522 and p['budget']['planned_cases']==1522
    permutation=np.random.default_rng(p['seed_bases']['permutation']).permutation(eligible).tolist()
    schedule=dict(permutation_seed=p['seed_bases']['permutation'],eligible_candidates=eligible,permutation=permutation,
                  cases=1522,truth_permutation_draws=1,individual_truth_draws=0,without_replacement=True,policy_access=False)
    event=ledger.append('truth_schedule',schedule)
    save(folder/'truth-schedule.json',dict(schedule,schedule_sequence=event['sequence'],schedule_record_sha256=event['sha256']))
    schedule_hash=event['sha256'];lookup={bytes(row):i for i,row in enumerate(ordinary)}
    fresh,truth_counts,global_boards={},{},set()
    sources=[] if sources is None else sources
    assert not sources
    for case,truth in enumerate(permutation):
        seed=p['seed_bases']['source']+case;population=sourcepop[truth]
        assert truth not in truth_counts
        indices=np.random.default_rng(seed).choice(len(population),8,replace=False);sample=population[indices]
        initial=np.flatnonzero(sourcepred[:,[lookup[bytes(row)] for row in sample]].all(1)).tolist()
        digest=hashlib.sha256(b''.join(sorted(map(bytes,sample)))).hexdigest()
        assert truth in initial and all(score_tables[role][c] is not None for role in ['H','T'] for c in initial)
        data=dict(case=case,candidate=truth,bank=0,truth_permutation_position=case,truth_schedule_record_sha256=schedule_hash,
            seed=seed,source_population=len(population),indices=indices.tolist(),columns_zero_based=sample.tolist(),
            initial_candidates=initial,unordered_sample_sha256=digest,prior_same_study_cases=list(fresh.get(digest,[])),
            historical_exact8_matches=history.get(digest,[]))
        event=ledger.append('source_bank',data)
        sources.append(dict(data,source_sequence=event['sequence'],source_record_sha256=event['sha256']))
        truth_counts[truth]=1;fresh.setdefault(digest,[]).append(case);global_boards.update(map(bytes,sample));statuses[case]['source']='complete'
    assert sorted(truth_counts)==eligible
    save(folder/'source-banks.json',sources)
    ledger.append('all_source_banks_drawn',dict(cases=1522,truth_permutation_draws=1,individual_truth_draws=0,new_source_draws=1522,
        source_positive_union_appearances=12176,positive_arm_appearances=24352,distinct_N12_boards=len(global_boards),
        source_banks_sha256=sha(folder/'source-banks.json')))
    return sources,truth_counts,global_boards
