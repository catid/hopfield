"""Separate uniform sampling, fixed fitted weights and expanded seed checks."""
import math
import numpy as np
from common import ROOT, SRC, read, save, sha, now
from prior import sampling_prior_rows, fitted_prior_rows, truth_from_ticket
from fractions import Fraction
from weights import integer_weights

TRUTH_BASE = 2027500000
SOURCE_BASE = 2027600000
CASES = 1024
# Supersets deliberately include ineligible/unselected candidate IDs and banks.
EXPANSIONS = [
    ('h3ax', 2026116000, 4*1541, '2026116000+4*family+bank',
     'conservative family 0..1540, bank 0..3'),
    ('h3be', 2026120000, 1541, '2026120000+candidateID',
     'candidateID 0..1540, including ineligible rows'),
    ('h3bf', 2026200000, 64*1541, '2026200000+64*candidateID+bank',
     'candidateID 0..1540, bank 0..63, including unused combinations'),
    ('h3bj', 2026300000, 64*1541, '2026300000+64*candidateID+bank',
     'candidateID 0..1540, bank 0..63, including unused combinations'),
    ('h3bm', 2026400000, 64*1541, '2026400000+64*candidateID+bank',
     'candidateID 0..1540, bank 0..63, including unused combinations'),
    ('h3bm', 2026500000, 64*1541, '2026500000+64*candidateID+bank',
     'candidateID 0..1540, bank 0..63, including unused combinations'),
    ('h3bq', 2026600000, 64*1541, '2026600000+64*candidateID+bank',
     'candidateID 0..1540, bank 0..63, including unused combinations'),
    ('h3br', 2026700000, 4096, '2026700000+case', 'case 0..4095'),
    ('h3br', 2026800000, 4096, '2026800000+case', 'case 0..4095'),
    ('h3bx', 2026900000, 4096, '2026900000+case', 'case 0..4095'),
    ('h3bx', 2027000000, 4096, '2027000000+case', 'case 0..4095'),
    ('h3by', 2027100000, 4096, '2027100000+case', 'case 0..4095'),
    ('h3by', 2027200000, 4096, '2027200000+case', 'case 0..4095'),
    ('h3bz', 2027300000, 1024, '2027300000+case', 'case 0..1023'),
    ('h3bz', 2027400000, 1024, '2027400000+case', 'case 0..1023'),
]


def run(folder, historical_protocols=None):
    folder.mkdir(parents=True, exist_ok=True)
    normalizers = read(SRC.parent/'h3br/protocol.json')['normalizers']
    selection = read(ROOT/normalizers['selection_path'])
    packed = np.load(ROOT/normalizers['bits_path'])['bits']
    counts = np.unpackbits(packed, axis=1, count=14200, bitorder='little').sum(1)
    rows = sampling_prior_rows(counts)
    fitted = fitted_prior_rows(counts)
    assert all(row['candidate'] == old['candidate'] and row['orbits'] == old['orbits']
               and row['source_population'] == old['complete_source_cardinality']
               for row, old in zip(rows, selection['candidates']))
    mapped = [truth_from_ticket(rows, ticket) for ticket in range(1522)]
    multiplicities = np.bincount(mapped, minlength=1541).tolist()
    assert sum(r['sampling_integer_weight'] for r in rows) == 1522
    assert multiplicities == [r['sampling_integer_weight'] for r in rows]
    assert sum(r['eligible'] for r in rows) == 1522
    assert fitted == read(SRC.parent/'h3br/protocol.json')['truths']
    assert sum(r['prior_integer_weight'] for r in fitted) == 4436
    save(folder/'prior-gate.json', dict(utc=now(), prior_rows=rows,
         fitted_prior_rows=fitted, eligible_models=1522, ineligible_models=19,
         sampling_total_integer_tickets=1522, fitted_total_integer_mass=4436,
         all1522_sampling_tickets_enumerated=True, ticket_to_candidate=mapped,
         model_ticket_multiplicities=multiplicities,
         source_selection_sha256=sha(ROOT/normalizers['selection_path']),
         source_bits_sha256=sha(ROOT/normalizers['bits_path']),
         fresh_truth_draws=0, fresh_source_draws=0))
    scores = read(ROOT/'results/research/loki/h3bm/score-values8.json')
    for row, score in zip(fitted, scores):
        M, k = row['source_population'], len(row['orbits'])
        expected = str(math.comb(55, k)*math.comb(M, 8)) if M >= 8 else None
        assert score == expected
    candidates = [next(row['candidate'] for row in fitted if row['eligible'] and len(row['orbits']) == k) for k in [0,1,2]]
    weights, weight_receipt = integer_weights(candidates, scores, 100000)
    fitted_mass = [Fraction(1, int(scores[c])) for c in candidates]
    actual_mass = [Fraction(1, math.comb(int(counts[c]), 8)) for c in candidates]
    assert all(Fraction(weights[i], weights[0]) == fitted_mass[i]/fitted_mass[0] for i in range(3))
    assert any(fitted_mass[i]/fitted_mass[0] != actual_mass[i]/actual_mass[0] for i in range(1,3))
    save(folder/'fitted-weight-gate.json', dict(candidates=candidates, integer_weight_receipt=weight_receipt,
         all1541_fitted_source_scores_unchanged=True,
         fitted_mass=[str(x) for x in fitted_mass], actual_uniform_prior_mass=[str(x) for x in actual_mass],
         fitted_and_actual_posteriors_explicitly_different=True,
         fixture_scope='Formal source-likelihood score ratios; no claim these three models share a realized source bank',
         fresh_truth_draws=0,fresh_source_draws=0,actual_oracle_calls=0))
    uniform_scores = [str(math.comb(int(M), 8)) if M >= 8 else None for M in counts]
    assert all((a is None) == (b is None) == (int(M) < 8) for a, b, M in zip(scores, uniform_scores, counts))
    uniform_weights, uniform_receipt = integer_weights(candidates, uniform_scores, 100000)
    assert all(Fraction(uniform_weights[i], uniform_weights[0]) == actual_mass[i]/actual_mass[0] for i in range(3))
    save(folder/'two-score-gate.json', dict(H_scores=scores, T_scores=uniform_scores, selector_scores=dict(H=scores,T=scores),
         H_fixture=weight_receipt, T_fixture=uniform_receipt, candidates=candidates,
         exactly_same1541_source_compatibility_support=True, ineligible_candidates=[i for i, M in enumerate(counts) if M < 8],
         ineligible_support_reason='Fewer than8 total N12 positives cannot contain any bank of8 distinct positives; initial candidates are always filtered across all1541 models',
         actual_uniform_mass=[str(x) for x in actual_mass], fitted_hierarchical_mass=[str(x) for x in fitted_mass],
         fresh_truth_draws=0, fresh_source_draws=0, actual_oracle_calls=0))
    old = {}
    def walk(value, path, context=''):
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, path, context+'/'+key)
        elif isinstance(value, list):
            for child in value:
                walk(child, path, context)
        elif type(value) is int and 'seed' in context.lower():
            old.setdefault(value, []).append(str(path.relative_to(ROOT))+':'+context)
    paths = (sorted(ROOT.glob('research/**/*protocol*.json')) if historical_protocols is None
             else [ROOT/path for path in historical_protocols])
    paths = [p for p in paths if SRC not in p.parents]
    identities = {}
    for path in paths:
        identities[str(path.relative_to(ROOT))] = sha(path)
        walk(read(path), path)
    expanded = set()
    ranges = []
    for study, base, count, formula, scope in EXPANSIONS:
        path = SRC.parent/study/'protocol.json'
        assert str(path.relative_to(ROOT)) in identities and str(base) in path.read_text()
        values = set(range(base, base+count))
        assert len(values) == count
        expanded.update(values)
        ranges.append(dict(protocol=str(path.relative_to(ROOT)), formula=formula,
                           first=base, last=base+count-1, seeds=count, scope=scope))
    for row in read(SRC/'proposed-seed-audit.json')['expanded_nearby_other_machine_solver_ranges']:
        path = ROOT/row['protocol']
        assert str(path.relative_to(ROOT)) in identities and sha(path) == row['protocol_sha256']
        expanded.update(range(row['first'], row['last']+1))
        ranges.append(row)
    prospective = [seed for i in range(CASES) for seed in [TRUTH_BASE+i, SOURCE_BASE+i]]
    explicit_collisions = {str(seed): old[seed] for seed in prospective if seed in old}
    expanded_collisions = sorted(set(prospective) & expanded)
    assert len(prospective) == len(set(prospective)) == 2048
    assert not explicit_collisions and not expanded_collisions
    receipt = dict(utc=now(), truth_seed_base=TRUTH_BASE, source_seed_base=SOURCE_BASE,
                   cases=CASES, prospective_seed_count=2048,
                   historical_protocol_sha256=identities,
                   historical_explicit_seed_count=len(old),
                   expanded_known_truth_source_ranges=ranges,
                   expanded_seed_union_count=len(expanded),
                   explicit_collisions=explicit_collisions,
                   expanded_collisions=expanded_collisions,
                   scope='Explicit integer seed fields across all protocol/proposal-protocol files, declared Loki truth/source formulas and three nearby Ripper solver formulas, with conservative unused combinations; no claim of exhaustive unexpanded streams or PRNG independence',
                   fresh_truth_draws=0, fresh_source_draws=0, actual_oracle_calls=0)
    save(folder/'seed-audit.json', receipt)
    return dict(prior_gate_sha256=sha(folder/'prior-gate.json'),
                fitted_weight_gate_sha256=sha(folder/'fitted-weight-gate.json'),
                two_score_gate_sha256=sha(folder/'two-score-gate.json'), H_scores=scores, T_scores=uniform_scores,
                seed_audit_sha256=sha(folder/'seed-audit.json'),
                prior_rows=rows, fitted_prior_rows=fitted, seed_audit=receipt)


if __name__ == '__main__':
    result = run(SRC)
    print({key: value for key, value in result['seed_audit'].items()
           if key != 'historical_protocol_sha256'})
