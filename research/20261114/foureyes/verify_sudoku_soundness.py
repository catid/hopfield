"""Public-only independent checker of clique/count proofs and counterexamples.

Imports neither the producer nor the training/search implementation.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PUBLIC = ROOT/'results/research/foureyes/logical_soundness'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_trace(support, trace):
    # Coordinate predicates independently reconstruct the 27 official units.
    units = [[x for x in range(81) if x//9 == k] for k in range(9)]
    units += [[x for x in range(81) if x % 9 == k] for k in range(9)]
    units += [[x for x in range(81) if (x//27)*3+(x % 9)//3 == k] for k in range(9)]
    pairs = [(i, j) for i in range(81) for j in range(i+1, 81)]
    edges = {p for p, present in zip(pairs, support) if present}
    proved = set()
    count_steps = 0
    for step in trace:
        unit = step['unit']
        assert isinstance(unit, int) and 0 <= unit < 27 and unit not in proved
        cells = units[unit]
        clique = {(min(i, j), max(i, j)) for i in cells for j in cells if i != j}
        if step['rule'] == 'clique':
            assert clique <= edges
        else:
            assert step['rule'] == 'count_identity'
            source, others = step['source'], step['others']
            assert all(i in proved for i in source+others)
            assert len(source) == len(others)+1
            lhs = Counter(x for i in source for x in units[i])
            rhs = Counter(x for i in others for x in units[i]) + Counter(cells)
            assert lhs == rhs
            count_steps += 1
        proved.add(unit)
        edges.update(clique)
    return proved, count_steps, units


def verify():
    result = json.loads((PUBLIC/'certificates.json').read_text())
    parent_path = ROOT/'results/research/foureyes/noisy_pl_long/summary.json'
    assert sha(parent_path) == result['parent_sha256']
    parent = json.loads(parent_path.read_text())
    assert sha(PUBLIC/'supports.npz') == result['supports_sha256']
    with np.load(PUBLIC/'supports.npz', allow_pickle=False) as data:
        supports = dict(zip(map(str, data['job_ids']), data['supports']))
    assert len(supports) == 121
    witnesses = {row['job']: row for row in result['witnesses']}
    pairlist = [(i, j) for i in range(81) for j in range(i+1, 81)]
    checked = []; count_steps = 0; rejected_mutations = 0
    for row, mapping in zip(result['rows'], parent['model_mappings'], strict=True):
        assert (row['bank'], row['arm'], row['job']) == (mapping['bank'], mapping['arm'], mapping['actual'])
        support = supports[row['job']]
        assert support.dtype == np.bool_ and support.shape == (3240,)
        assert hashlib.sha256(support.tobytes()).hexdigest() == mapping['support_sha256'] == row['support_sha256']
        proved, steps, units = check_trace(support, row['trace'])
        assert sorted(proved) == row['certified_units'] and row['sound'] == (len(proved) == 27)
        count_steps += steps
        outcome = next(r for r in parent['logical_bank_outcomes'] if (r['bank'], r['arm']) == (row['bank'], row['arm']))
        assert row['observed_wrong_stops'] == outcome['wrong_stops']
        if row['sound']:
            assert outcome['wrong_stops'] == 0 and 'witness_job' not in row
        elif 'witness_job' in row:
            witness = witnesses[row['witness_job']]
            assert witness['job'] == row['job']
            assert parent['input_sha256'][witness['source_archive']] == witness['source_sha256']
            board, solution = np.array(witness['board']), np.array(witness['solution'])
            free = np.array(witness['free'])
            assert board.shape == solution.shape == free.shape == (81,) and free.dtype == np.bool_
            assert np.issubdtype(board.dtype, np.integer) and np.issubdtype(solution.dtype, np.integer)
            assert ((board >= 0) & (board < 9)).all()
            assert np.array_equal(board[~free], solution[~free])
            assert all(set(solution[cells]) == set(range(9)) for cells in units)
            assert any(len(set(board[cells])) != 9 for cells in units)
            assert all(board[i] != board[j] for (i, j), keep in zip(pairlist, support) if keep)
            assert witness['puzzle_index'] == 260000+witness['local_index']
        if steps:
            # A bogus count identity must be rejected, not trusted by its label.
            import copy
            broken = copy.deepcopy(row['trace'])
            first = next(s for s in broken if s['rule'] == 'count_identity')
            first['source'] = first['source'][:-1]
            try:
                check_trace(support, broken)
            except AssertionError:
                rejected_mutations += 1
            else:
                raise AssertionError('accepted invalid count identity')
        checked.append(row)
    for group in result['grouped']:
        rows = [r for r in checked if r['arm'] == group['arm']]
        assert len(rows) == 64
        assert sum(r['sound'] for r in rows) == group['sound']
        assert sum('witness_job' in r for r in rows) == group['unsound_witness']
        assert sum(not r['sound'] and 'witness_job' not in r for r in rows) == group['unresolved']
    return dict(status='passed', logical_models=len(checked), unique_models=len(supports),
                count_identity_steps=count_steps, rejected_bad_proofs=rejected_mutations,
                grouped=result['grouped'], verifier_sha256=sha(__file__))


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
