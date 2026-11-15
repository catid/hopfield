"""Exhaustive prior and expanded seed-range checks without prospective draws."""
import math
import numpy as np
from common import ROOT, SRC, read, save, sha, now
from prior import prior_rows, truth_from_ticket

TRUTH_BASE = 2026900000
SOURCE_BASE = 2027000000
CASES = 4096
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
]


def run(folder, historical_protocols=None):
    folder.mkdir(parents=True, exist_ok=True)
    normalizers = read(SRC.parent/'h3br/protocol.json')['normalizers']
    selection = read(ROOT/normalizers['selection_path'])
    packed = np.load(ROOT/normalizers['bits_path'])['bits']
    counts = np.unpackbits(packed, axis=1, count=14200, bitorder='little').sum(1)
    rows = prior_rows(counts)
    assert all(row['candidate'] == old['candidate'] and row['orbits'] == old['orbits']
               and row['source_population'] == old['complete_source_cardinality']
               for row, old in zip(rows, selection['candidates']))
    mapped = [truth_from_ticket(rows, ticket) for ticket in range(4436)]
    multiplicities = np.bincount(mapped, minlength=1541).tolist()
    assert sum(r['prior_integer_weight'] for r in rows) == 4436
    assert multiplicities == [r['prior_integer_weight'] for r in rows]
    assert sum(r['eligible'] for r in rows) == 1522
    assert rows == read(SRC.parent/'h3br/protocol.json')['truths']
    save(folder/'prior-gate.json', dict(utc=now(), prior_rows=rows,
         eligible_models=1522, ineligible_models=19, total_integer_tickets=4436,
         all4436_tickets_enumerated=True, ticket_to_candidate=mapped,
         model_ticket_multiplicities=multiplicities,
         source_selection_sha256=sha(ROOT/normalizers['selection_path']),
         source_bits_sha256=sha(ROOT/normalizers['bits_path']),
         fresh_truth_draws=0, fresh_source_draws=0))
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
    paths = (sorted(ROOT.glob('research/**/protocol.json')) if historical_protocols is None
             else [ROOT/path for path in historical_protocols])
    paths = [p for p in paths if p != SRC/'protocol.json']
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
    prospective = [seed for i in range(CASES) for seed in [TRUTH_BASE+i, SOURCE_BASE+i]]
    explicit_collisions = {str(seed): old[seed] for seed in prospective if seed in old}
    expanded_collisions = sorted(set(prospective) & expanded)
    assert len(prospective) == len(set(prospective)) == 8192
    assert not explicit_collisions and not expanded_collisions
    receipt = dict(utc=now(), truth_seed_base=TRUTH_BASE, source_seed_base=SOURCE_BASE,
                   cases=CASES, prospective_seed_count=8192,
                   historical_protocol_sha256=identities,
                   historical_explicit_seed_count=len(old),
                   expanded_known_truth_source_ranges=ranges,
                   expanded_seed_union_count=len(expanded),
                   explicit_collisions=explicit_collisions,
                   expanded_collisions=expanded_collisions,
                   scope='Explicit integer seed fields plus declared prior truth/source formula ranges, including conservative unused candidate/bank combinations; no claim about undocumented RNG streams',
                   fresh_truth_draws=0, fresh_source_draws=0, actual_oracle_calls=0)
    save(folder/'seed-audit.json', receipt)
    return dict(prior_gate_sha256=sha(folder/'prior-gate.json'),
                seed_audit_sha256=sha(folder/'seed-audit.json'),
                prior_rows=rows, seed_audit=receipt)


if __name__ == '__main__':
    result = run(SRC)
    print({key: value for key, value in result['seed_audit'].items()
           if key != 'historical_protocol_sha256'})
