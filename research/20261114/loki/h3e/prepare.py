"""Independent n=12 universe, disjoint input splits and exact PL sufficient statistics."""
import datetime as dt
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / 'research/20261114/loki/h3e'
OUT = ROOT / 'artifacts/research/20261114/loki/h3e'
sys.path.insert(0, str(ROOT / 'upstream'))
from cram import nqueens_data as D


def sha(path):
    with Path(path).open('rb') as handle: return hashlib.file_digest(handle, 'sha256').hexdigest()


def bitmask_solutions(n):
    """Independent row recursion using occupied column and shifted diagonal bitsets."""
    all_bits = (1 << n) - 1
    answers = []
    def visit(columns, left, right, chosen):
        if columns == all_bits:
            answers.append(chosen); return
        available = all_bits & ~(columns | left | right)
        while available:
            bit = available & -available; available ^= bit
            visit(columns | bit, ((left | bit) << 1) & all_bits, (right | bit) >> 1,
                  chosen + [bit.bit_length() - 1])
    visit(0, 0, 0, [])
    return np.asarray(answers, np.int16)


def decode(keys, c=144):
    digits = np.stack([keys // 145**2, keys // 145 % 145, keys % 145], axis=1)
    boards = np.zeros((len(keys), c), bool)
    for j in range(3):
        keep = digits[:, j] > 0; boards[np.flatnonzero(keep), digits[keep, j]-1] = True
    return boards


def allocations(sizes, count):
    ideal = np.asarray(sizes, float) / np.sum(sizes) * count
    result = np.floor(ideal).astype(int)
    for i in np.argsort(-(ideal-result))[:count-int(result.sum())]: result[i] += 1
    return result


def main():
    started = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'dataset.json').exists(), 'Dataset is frozen'
    n = 12; cols = bitmask_solutions(n); assert cols.shape == (14200, 12)
    positions = (np.arange(n)[None, :] * n + cols).astype(np.int16)
    solutions = np.zeros((len(cols), n*n), bool); solutions[np.arange(len(cols))[:, None], positions] = True
    # Validate rows, columns and diagonals without either generator's recursion.
    assert np.all(np.sort(cols, axis=1) == np.arange(n))
    assert all(len(set((np.arange(n)-row).tolist())) == n and len(set((np.arange(n)+row).tolist())) == n for row in cols)
    assert len(set(map(bytes, solutions))) == 14200
    reference = np.asarray(D.completions(np.zeros(n*n, bool), n), bool)
    assert {row.tobytes() for row in solutions} == {row.tobytes() for row in reference}
    keys_all = []; solution_all = []
    for k in [1, 2, 3]:
        subsets = np.asarray(list(itertools.combinations(range(n), k)), np.int16)
        cells = positions[:, subsets] + 1
        keys = sum(cells[:, :, j].astype(np.int64) * 145**(2-j) for j in range(k))
        keys_all.append(keys.ravel()); solution_all.append(np.repeat(np.arange(len(cols), dtype=np.int32), len(subsets)))
    keys_all = np.concatenate(keys_all); solution_all = np.concatenate(solution_all)
    assert len(keys_all) == 4231600
    unique, inverse, multiplicities = np.unique(keys_all, return_inverse=True, return_counts=True)
    clues = (unique // 145**2 > 0).astype(np.int8) + (unique // 145 % 145 > 0) + (unique % 145 > 0)
    strata = [np.flatnonzero(clues == k) for k in [1, 2, 3]]
    assert sum(map(len, strata)) == len(unique) >= 108192
    held_sizes = allocations(list(map(len, strata)), 8192)
    fit_sizes = allocations(np.array(list(map(len, strata))) - held_sizes, 100000)
    rng = np.random.default_rng(2026111412)
    held = []; fit = []
    for values, h, f in zip(strata, held_sizes, fit_sizes):
        order = rng.permutation(values)
        assert h+f <= len(values)
        held.extend(order[:h]); fit.extend(order[h:h+f])
    held = np.sort(held); fit = np.sort(fit)
    assert len(set(held) & set(fit)) == 0
    status = np.zeros(len(unique), np.int8); status[fit] = 1; status[held] = 2
    included = status[inverse] == 1
    train_solution = solution_all[included]; train_keys = keys_all[included]
    rows_per_solution = np.bincount(train_solution, minlength=len(solutions))
    given_counts = np.zeros((len(solutions), n*n), np.int64)
    for power in [2, 1, 0]:
        cells = train_keys // 145**power % 145; present = cells > 0
        np.add.at(given_counts, (train_solution[present], cells[present]-1), 1)
    free_counts = rows_per_solution[:, None] - given_counts
    assert np.all(free_counts >= 0) and np.all(rows_per_solution > 0)
    fit_inputs = decode(unique[fit]); held_inputs = decode(unique[held])
    # Sparse labels are all compatible complete solutions, exactly the GRAM row expansion.
    assert np.all(solutions[train_solution[np.arange(0, len(train_solution), 1000)]] >= decode(train_keys[::1000]))
    np.savez_compressed(OUT / 'data.npz', solutions=solutions, solution_columns=cols,
        fit_keys=unique[fit], fit_inputs=fit_inputs, fit_solution_counts=multiplicities[fit],
        test_keys=unique[held], test_inputs=held_inputs, test_solution_counts=multiplicities[held],
        free_counts=free_counts, rows_per_solution=rows_per_solution,
        expanded_fit_solution_indices=train_solution, expanded_fit_keys=train_keys,
        full_universe_keys=unique, full_universe_solution_counts=multiplicities, full_universe_clues=clues)
    report = dict(status='prepared_and_independently_validated', utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        n=n, complete_solutions=14200, exact_labeled_rows_universe=len(keys_all), unique_input_universe=len(unique),
        full_unique_by_clues=dict(zip(['1', '2', '3'], map(len, strata))),
        fit_unique=100000, fit_unique_by_clues=dict(zip(['1', '2', '3'], fit_sizes.tolist())),
        test_unique=8192, test_unique_by_clues=dict(zip(['1', '2', '3'], held_sizes.tolist())),
        fit_expanded_rows=int(included.sum()), test_expanded_rows=int(multiplicities[held].sum()),
        free_cell_weight_total=int(free_counts.sum()), seed=2026111412,
        splitting='Enumerate every completable 1/2/3-clue input; allocate 8192 held-out inputs proportionally by clue count using largest remainders, shuffle within each stratum, take held-out first. Allocate 100000 fit inputs proportionally from remaining strata and take next shuffled inputs. Remainder unused. No partial input crosses groups.',
        labels='Every compatible complete solution for each selected fit input, matching GRAM expansion; free_counts aggregates the exact full-row PL objective without sampling a completion.',
        validation=['Independent bitmask recursion yields14200 distinct solutions', 'Direct column/diagonal checks on every solution', 'Exact set equality against pinned set-based completion recursion', 'Input-group split disjointness and exact target counts', 'All selected completion labels represented in exact free-cell counts'],
        data_sha256=sha(OUT / 'data.npz'), source_sha256=sha(__file__), seconds=time.perf_counter()-started,
        no_training_or_search=True)
    (OUT / 'dataset.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__': main()
