"""Independent reconstruction of all fresh prior tickets and source choices."""
from bisect import bisect_right
import hashlib
import itertools
import math
import numpy as np


def verify(root, folder, protocol, sources, ledger, read, sha, check):
    p = protocol
    ordinary = np.load(root/p['normalizers']['ordinary_path'])
    packed = np.load(root/p['normalizers']['bits_path'])['bits']
    pred = np.unpackbits(packed, axis=1, count=len(ordinary), bitorder='little').astype(bool)
    selection = read(root/p['normalizers']['selection_path'])
    orbits = list(itertools.combinations(range(1, 12), 2))
    choices = [chosen for k in [0, 1, 2] for chosen in itertools.combinations(range(55), k)]
    prior = []
    for candidate, chosen in enumerate(choices):
        M = int(pred[candidate].sum())
        row = dict(candidate=candidate, orbits=[list(orbits[j]) for j in chosen],
                   source_population=M, eligible=M >= 8,
                   prior_integer_weight=1485//math.comb(55, len(chosen)) if M >= 8 else 0)
        prior.append(row)
        old = selection['candidates'][candidate]
        check('full_source_population_and_class', row['candidate'] == old['candidate'] and
              row['orbits'] == old['orbits'] and M == old['complete_source_cardinality'])
    cumulative, total = [], 0
    for row in prior:
        total += row['prior_integer_weight']
        cumulative.append(total)
    counts = [0]*1541
    for ticket in range(total):
        counts[bisect_right(cumulative, ticket)] += 1
    check('all_exact_prior_tickets', total == 4436 and prior == p['truths'] and
          counts == [row['prior_integer_weight'] for row in prior] and sum(r['eligible'] for r in prior) == 1522)
    prior_gate = read(folder/'source-gates/prior-gate.json')
    check('runtime_prior_gate_identity', prior_gate['prior_rows'] == prior and
          prior_gate['ticket_to_candidate'] == [bisect_right(cumulative, t) for t in range(total)] and
          prior_gate['model_ticket_multiplicities'] == counts and
          prior_gate['source_selection_sha256'] == sha(root/p['normalizers']['selection_path']) and
          prior_gate['source_bits_sha256'] == sha(root/p['normalizers']['bits_path']))
    seeds = read(folder/'source-gates/seed-audit.json')
    explicit = set()
    def walk(value, context=''):
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, context+'/'+key)
        elif isinstance(value, list):
            for child in value:
                walk(child, context)
        elif type(value) is int and 'seed' in context.lower():
            explicit.add(value)
    check('historical_seed_protocols_bound', seeds['historical_protocol_sha256'] == p['historical_protocol_sha256'])
    for relative, digest in p['historical_protocol_sha256'].items():
        check('historical_seed_protocol_hash', sha(root/relative) == digest)
        walk(read(root/relative))
    proposed = {base+i for base in p['seed_bases'].values() for i in range(4096)}
    check('distinct_new_explicit_seeds', len(proposed) == 8192 and not (proposed & explicit) and
          len(explicit) == seeds['historical_explicit_seed_count'] and not seeds['explicit_collisions'])
    expected = [('h3ax', 2026116000, 6164), ('h3be', 2026120000, 1541),
                ('h3bf', 2026200000, 98624), ('h3bj', 2026300000, 98624),
                ('h3bm', 2026400000, 98624), ('h3bm', 2026500000, 98624),
                ('h3bq', 2026600000, 98624), ('h3br', 2026700000, 4096), ('h3br', 2026800000, 4096)]
    check('all_known_ranges_retained', len(seeds['expanded_known_truth_source_ranges']) == len(expected))
    expanded = set()
    for row, (study, base, count) in zip(seeds['expanded_known_truth_source_ranges'], expected):
        check('known_seed_range_bounds', row['protocol'] == f'research/20261114/loki/{study}/protocol.json' and
              row['first'] == base and row['last'] == base+count-1 and row['seeds'] == count)
        expanded.update(range(base, base+count))
    check('all_expanded_seed_ranges_disjoint', not (proposed & expanded) and
          len(expanded) == seeds['expanded_seed_union_count'] == 507476 and not seeds['expanded_collisions'])
    return verify_banks(root, folder, p, sources, ledger, read, sha, check, ordinary, pred, cumulative)


def verify_banks(root, folder, p, sources, ledger, read, sha, check, ordinary, pred, cumulative):
    history = {}
    for row in read(root/'results/research/loki/h3bm/source-pairs.json'):
        history.setdefault(row['unordered_sample_sha256']['8'], []).append(dict(study='H3bm', case=row['case'], candidate=row['candidate']))
    for study, path in [('H3bq', 'results/research/loki/h3bq/source-banks.json'),
                        ('H3br', 'results/research/loki/h3br/source-banks.json')]:
        for row in read(root/path):
            history.setdefault(row['unordered_sample_sha256'], []).append(dict(study=study, case=row['case'], candidate=row['candidate']))
    check('complete_historical_duplicate_registry', read(folder/'historical-eight-banks.json') == history)
    lookup = {bytes(row): i for i, row in enumerate(ordinary)}
    fresh, truth_counts, global_boards, initials = {}, {}, set(), []
    check('all4096_sources_retained', len(sources) == 4096 and [s['case'] for s in sources] == list(range(4096)))
    for case, source in enumerate(sources):
        truth_seed, seed = p['seed_bases']['truth']+case, p['seed_bases']['source']+case
        ticket = int(np.random.default_rng(truth_seed).integers(4436))
        truth = bisect_right(cumulative, ticket)
        population = ordinary[pred[truth]]
        indices = np.random.default_rng(seed).choice(len(population), 8, replace=False)
        sample = population[indices]
        selected = [lookup[bytes(row)] for row in sample]
        initial = np.flatnonzero(pred[:, selected].all(1)).tolist()
        digest = hashlib.sha256(b''.join(sorted(map(bytes, sample)))).hexdigest()
        bank = truth_counts.get(truth, 0)
        data = dict(case=case, candidate=truth, bank=bank, truth_seed=truth_seed,
                    truth_ticket=ticket, seed=seed, source_population=len(population),
                    indices=indices.tolist(), columns_zero_based=sample.tolist(),
                    initial_candidates=initial, unordered_sample_sha256=digest,
                    prior_same_study_cases=list(fresh.get(digest, [])),
                    historical_exact8_matches=history.get(digest, []))
        event = ledger.take('source_bank')
        check('fresh_truth_source_rng_and_full_filter', event['data'] == data and
              source == dict(data, source_sequence=event['sequence'], source_record_sha256=event['sha256']) and
              len(selected) == len(set(selected)) == 8 and truth in initial and pred[truth, selected].all())
        truth_counts[truth] = bank+1
        fresh.setdefault(digest, []).append(case)
        global_boards.update(map(bytes, sample))
        initials.append(initial)
    check('all_sources_before_any_target_query', ledger.take('all_source_banks_drawn')['data'] == dict(
          cases=4096, new_truth_draws=4096, new_source_draws=4096,
          source_positive_union_appearances=32768, positive_arm_appearances=65536,
          distinct_N12_boards=len(global_boards), source_banks_sha256=sha(folder/'source-banks.json')))
    return dict(initials=initials, truth_counts=truth_counts, distinct_N12_boards=len(global_boards))
