"""Independent reconstruction of all fresh prior tickets and source choices."""
from bisect import bisect_right
import hashlib
import itertools
import math
from fractions import Fraction
import numpy as np


def verify(root, folder, protocol, sources, ledger, read, sha, check):
    ordinary, pred, cumulative = verify_prior(root, folder, protocol, read, sha, check)
    return verify_banks(root, folder, protocol, sources, ledger, read, sha, check, ordinary, pred, cumulative)


def verify_prior(root, folder, p, read, sha, check):
    ordinary = np.load(root/p['normalizers']['ordinary_path'])
    packed = np.load(root/p['normalizers']['bits_path'])['bits']
    pred = np.unpackbits(packed, axis=1, count=len(ordinary), bitorder='little').astype(bool)
    selection = read(root/p['normalizers']['selection_path'])
    orbits = list(itertools.combinations(range(1, 12), 2))
    choices = [chosen for k in [0, 1, 2] for chosen in itertools.combinations(range(55), k)]
    prior, fitted = [], []
    for candidate, chosen in enumerate(choices):
        M = int(pred[candidate].sum())
        row = dict(candidate=candidate, orbits=[list(orbits[j]) for j in chosen],
                   source_population=M, eligible=M >= 8,
                   sampling_integer_weight=int(M >= 8))
        prior.append(row)
        fitted.append(dict(candidate=candidate, orbits=row['orbits'], source_population=M, eligible=M >= 8,
                           prior_integer_weight=1485//math.comb(55, len(chosen)) if M >= 8 else 0))
        old = selection['candidates'][candidate]
        check('full_source_population_and_class', row['candidate'] == old['candidate'] and
              row['orbits'] == old['orbits'] and M == old['complete_source_cardinality'])
    cumulative, total = [], 0
    for row in prior:
        total += row['sampling_integer_weight']
        cumulative.append(total)
    counts = [0]*1541
    for ticket in range(total):
        counts[bisect_right(cumulative, ticket)] += 1
    check('all_exact_prior_tickets', total == 1522 and prior == p['truths'] and
          counts == [row['sampling_integer_weight'] for row in prior] and sum(r['eligible'] for r in prior) == 1522)
    check('separate_fixed_fitted_hierarchy', fitted == p['fitted_prior_rows'] ==
          read(root/'research/20261114/loki/h3br/protocol.json')['truths'] and
          sum(row['prior_integer_weight'] for row in fitted) == 4436)
    prior_gate = read(folder/'source-gates/prior-gate.json')
    check('runtime_prior_gate_identity', prior_gate['prior_rows'] == prior and prior_gate['fitted_prior_rows'] == fitted and
          prior_gate['sampling_total_integer_tickets'] == 1522 and prior_gate['fitted_total_integer_mass'] == 4436 and
          prior_gate['all1522_sampling_tickets_enumerated'] and
          prior_gate['fresh_truth_draws'] == prior_gate['fresh_source_draws'] == 0 and
          prior_gate['ticket_to_candidate'] == [bisect_right(cumulative, t) for t in range(total)] and
          prior_gate['model_ticket_multiplicities'] == counts and
          prior_gate['source_selection_sha256'] == sha(root/p['normalizers']['selection_path']) and
          prior_gate['source_bits_sha256'] == sha(root/p['normalizers']['bits_path']))
    scores = read(root/'results/research/loki/h3bm/score-values8.json')
    expected_scores = [str(math.comb(55, len(row['orbits']))*math.comb(row['source_population'], 8))
                       if row['eligible'] else None for row in fitted]
    check('all1541_fixed_fitted_source_scores', scores == expected_scores)
    candidates = [next(r['candidate'] for r in fitted if r['eligible'] and len(r['orbits']) == k) for k in [0, 1, 2]]
    denominators = [int(scores[c]) for c in candidates]
    common, intermediate = 1, []
    for denominator in denominators:
        common = math.lcm(common, denominator)
        intermediate.append(common.bit_length())
    weights = [common//d for d in denominators]
    fitted_mass = [Fraction(1, d) for d in denominators]
    actual_mass = [Fraction(1, math.comb(prior[c]['source_population'], 8)) for c in candidates]
    expected_weight_gate = dict(candidates=candidates, integer_weight_receipt=dict(
        score_denominators=[str(d) for d in denominators], common_denominator=str(common),
        LCM_intermediate_bit_lengths=intermediate, integer_weights=[str(w) for w in weights],
        weight_bit_lengths=[w.bit_length() for w in weights], max_bits=100000),
        all1541_fitted_source_scores_unchanged=True, fitted_mass=[str(x) for x in fitted_mass],
        actual_uniform_prior_mass=[str(x) for x in actual_mass],
        fitted_and_actual_posteriors_explicitly_different=True,
        fixture_scope='Formal source-likelihood score ratios; no claim these three models share a realized source bank',
        fresh_truth_draws=0, fresh_source_draws=0, actual_oracle_calls=0)
    check('independent_separate_fitted_weight_gate', read(folder/'source-gates/fitted-weight-gate.json') == expected_weight_gate and
          all(Fraction(weights[i], weights[0]) == fitted_mass[i]/fitted_mass[0] for i in range(3)) and
          any(fitted_mass[i]/fitted_mass[0] != actual_mass[i]/actual_mass[0] for i in [1, 2]))
    uniform_scores = [str(math.comb(row['source_population'],8)) if row['eligible'] else None for row in prior]
    uc, ui = 1, []
    for c in candidates:
        uc = math.lcm(uc,int(uniform_scores[c])); ui.append(uc.bit_length())
    uw = [uc//int(uniform_scores[c]) for c in candidates]
    ur = dict(score_denominators=[uniform_scores[c] for c in candidates],common_denominator=str(uc),
              LCM_intermediate_bit_lengths=ui,integer_weights=[str(x) for x in uw],
              weight_bit_lengths=[x.bit_length() for x in uw],max_bits=100000)
    expected_two_score_gate = dict(H_scores=scores,T_scores=uniform_scores,selector_scores=dict(H=scores,T=scores),
         H_fixture=expected_weight_gate['integer_weight_receipt'],T_fixture=ur,candidates=candidates,
         exactly_same1541_source_compatibility_support=True,ineligible_candidates=[r['candidate'] for r in prior if not r['eligible']],
         ineligible_support_reason='Fewer than8 total N12 positives cannot contain any bank of8 distinct positives; initial candidates are always filtered across all1541 models',
         actual_uniform_mass=[str(x) for x in actual_mass],fitted_hierarchical_mass=[str(x) for x in fitted_mass],
         fresh_truth_draws=0,fresh_source_draws=0,actual_oracle_calls=0)
    check('independent_uniform_and_hierarchical_scores_and_support',read(folder/'source-gates/two-score-gate.json') == expected_two_score_gate and
          all((h is None)==(u is None)==(not row['eligible']) for h,u,row in zip(scores,uniform_scores,prior)) and
          all(Fraction(uw[i],uw[0])==actual_mass[i]/actual_mass[0] for i in range(3)))
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
    proposed = {p['seed_bases']['permutation']} | {p['seed_bases']['source']+i for i in range(1522)}
    check('distinct_new_explicit_seeds', len(proposed) == 1523 and not (proposed & explicit) and
          len(explicit) == seeds['historical_explicit_seed_count'] and not seeds['explicit_collisions'])
    expected = [('h3ax', 2026116000, 6164), ('h3be', 2026120000, 1541),
                ('h3bf', 2026200000, 98624), ('h3bj', 2026300000, 98624),
                ('h3bm', 2026400000, 98624), ('h3bm', 2026500000, 98624),
                ('h3bq', 2026600000, 98624), ('h3br', 2026700000, 4096), ('h3br', 2026800000, 4096),
                ('h3bx', 2026900000, 4096), ('h3bx', 2027000000, 4096),
                ('h3by', 2027100000, 4096), ('h3by', 2027200000, 4096),
                ('h3bz', 2027300000, 1024), ('h3bz', 2027400000, 1024), ('h3ca',2027500000,1024), ('h3ca',2027600000,1024)]
    check('all_known_ranges_retained', len(seeds['expanded_known_truth_source_ranges']) == len(expected)+3)
    expanded = set()
    for row, (study, base, count) in zip(seeds['expanded_known_truth_source_ranges'], expected):
        check('known_seed_range_bounds', row['protocol'] == f'research/20261114/loki/{study}/protocol.json' and
              row['first'] == base and row['last'] == base+count-1 and row['seeds'] == count)
        expanded.update(range(base, base+count))
    for row, (name, base, grids) in zip(seeds['expanded_known_truth_source_ranges'][len(expected):],
            [('protocol-signal-detection.json',2027111502,4096),
             ('protocol-stability-stopping.json',2027311502,128),
             ('protocol-fixed-270-confirm.json',2027331502,128)]):
        rel = 'research/20261114/ripper/'+name
        check('nearby_other_machine_solver_ranges', row['protocol'] == rel and
              row['protocol_sha256'] == sha(root/rel) and row['first'] == base and
              row['last'] == base+41*grids-1 and row['seeds'] == 41*grids)
        expanded.update(range(base,base+41*grids))
    check('all_expanded_seed_ranges_disjoint', not (proposed & expanded) and
          len(expanded) == seeds['expanded_seed_union_count'] == 702292 and not seeds['expanded_collisions'])
    return ordinary, pred, cumulative


def verify_banks(root, folder, p, sources, ledger, read, sha, check, ordinary, pred, cumulative):
    history = {}
    for row in read(root/'results/research/loki/h3bm/source-pairs.json'):
        history.setdefault(row['unordered_sample_sha256']['8'], []).append(dict(study='H3bm', case=row['case'], candidate=row['candidate']))
    for study, path in [('H3bq', 'results/research/loki/h3bq/source-banks.json'),
                        ('H3br', 'results/research/loki/h3br/source-banks.json'),
                        ('H3bx', 'results/research/loki/h3bx/source-banks.json'),
                        ('H3by', 'results/research/loki/h3by/source-banks.json'),
                        ('H3bz', 'results/research/loki/h3bz/source-banks.json'),
                        ('H3ca', 'results/research/loki/h3ca/source-banks.json')]:
        for row in read(root/path):
            history.setdefault(row['unordered_sample_sha256'], []).append(dict(study=study, case=row['case'], candidate=row['candidate']))
    check('complete_historical_duplicate_registry', read(folder/'historical-eight-banks.json') == history)
    lookup = {bytes(row): i for i, row in enumerate(ordinary)}
    fresh, truth_counts, global_boards, initials = {}, {}, set(), []
    eligible=[r['candidate'] for r in p['truths'] if r['eligible']]
    permutation=np.random.default_rng(p['seed_bases']['permutation']).permutation(eligible).tolist()
    event=ledger.take('truth_schedule')
    expected_schedule=dict(permutation_seed=p['seed_bases']['permutation'],eligible_candidates=eligible,permutation=permutation,
        cases=1522,truth_permutation_draws=1,individual_truth_draws=0,without_replacement=True,policy_access=False)
    check('one_fresh_truth_permutation_full_eligible_coverage',event['data']==expected_schedule and read(folder/'truth-schedule.json')==dict(expected_schedule,
        schedule_sequence=event['sequence'],schedule_record_sha256=event['sha256']) and sorted(permutation)==eligible)
    schedule_hash=event['sha256']
    check('all1522_sources_retained',len(sources)==1522 and [s['case'] for s in sources]==list(range(1522)))
    for case, source in enumerate(sources):
        seed=p['seed_bases']['source']+case
        truth=permutation[case]
        population = ordinary[pred[truth]]
        indices = np.random.default_rng(seed).choice(len(population), 8, replace=False)
        sample = population[indices]
        selected = [lookup[bytes(row)] for row in sample]
        initial = np.flatnonzero(pred[:, selected].all(1)).tolist()
        digest = hashlib.sha256(b''.join(sorted(map(bytes, sample)))).hexdigest()
        bank = truth_counts.get(truth, 0)
        data = dict(case=case,candidate=truth,bank=bank,truth_permutation_position=case,truth_schedule_record_sha256=schedule_hash,
                    seed=seed,source_population=len(population),
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
          cases=1522, truth_permutation_draws=1, individual_truth_draws=0, new_source_draws=1522,
          source_positive_union_appearances=12176, positive_arm_appearances=24352,
          distinct_N12_boards=len(global_boards), source_banks_sha256=sha(folder/'source-banks.json')))
    check('exactly_one_bank_per_eligible_truth',truth_counts=={c:1 for c in eligible})
    return dict(initials=initials,truth_counts=truth_counts,distinct_N12_boards=len(global_boards))
