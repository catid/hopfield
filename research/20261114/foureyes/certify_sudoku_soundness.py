"""Post-hoc exact zero-set certificates for every frozen H4s support.

No fitting or search. Sudoku geometry is supplied only to this diagnostic.
"""
from pathlib import Path
from itertools import combinations, permutations
from datetime import datetime, timezone
import hashlib
import json
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / 'artifacts/research/20261114/foureyes/noisy_pl_long'
PARENT = ROOT / 'results/research/foureyes/noisy_pl_long'
PUBLIC = ROOT / 'results/research/foureyes/logical_soundness'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def units():
    return ([list(range(9*r, 9*r+9)) for r in range(9)]
            + [[9*r+c for r in range(9)] for c in range(9)]
            + [[9*r+c for r in range(3*a, 3*a+3) for c in range(3*b, 3*b+3)]
               for a in range(3) for b in range(3)])


def close(support):
    groups = units()
    pairlist = list(combinations(range(81), 2))
    edges = {pair for pair, keep in zip(pairlist, support) if keep}
    cliques = [set(combinations(sorted(group), 2)) for group in groups]
    partitions = [[list(range(9)), list(range(9, 18)), list(range(18, 27))]]
    partitions += [[[3*b+i for i in range(3)], [18+3*b+i for i in range(3)]] for b in range(3)]
    partitions += [[[9+3*b+i for i in range(3)], [18+3*i+b for i in range(3)]] for b in range(3)]
    known = set()
    trace = []
    while True:
        before = len(known)
        for i, clique in enumerate(cliques):
            if i not in known and clique <= edges:
                known.add(i)
                trace.append(dict(rule='clique', unit=i))
        for alternatives in partitions:
            for source, target in permutations(alternatives, 2):
                missing = set(target) - known
                if set(source) <= known and len(missing) == 1:
                    unit = missing.pop()
                    others = [i for i in target if i != unit]
                    trace.append(dict(rule='count_identity', unit=unit, source=source, others=others))
                    known.add(unit)
                    edges.update(cliques[unit])
        if len(known) == before:
            break
    return trace, sorted(known)


def main():
    start = time.perf_counter()
    summary = json.loads((PARENT / 'summary.json').read_text())
    prepared = json.loads((RAW / 'prepared.json').read_text())
    assert summary['input_sha256'][str((RAW/'prepared.json').relative_to(ROOT))] == sha(RAW/'prepared.json')
    PUBLIC.mkdir(parents=True, exist_ok=True)
    jobs = sorted(prepared['jobs'], key=lambda row: row['job_id'])
    supports = []
    for job in jobs:
        path = ROOT / job['model_path']
        assert sha(path) == job['model_sha256']
        support = np.load(path, allow_pickle=False)['support']
        assert support.dtype == np.bool_ and support.shape == (3240,)
        assert hashlib.sha256(support.tobytes()).hexdigest() == job['support_sha256']
        supports.append(support)
    np.savez_compressed(PUBLIC/'supports.npz', supports=np.stack(supports), job_ids=np.array([r['job_id'] for r in jobs]))
    witness_cache = {}
    rows = []
    for mapping in summary['model_mappings']:
        index = next(i for i, row in enumerate(jobs) if row['job_id'] == mapping['actual'])
        support = supports[index]
        trace, known = close(support)
        result = dict(bank=mapping['bank'], arm=mapping['arm'], job=mapping['actual'],
                      support_sha256=mapping['support_sha256'], trace=trace, certified_units=known,
                      sound=len(known) == 27)
        outcome = next(r for r in summary['logical_bank_outcomes'] if r['bank'] == mapping['bank'] and r['arm'] == mapping['arm'])
        result['observed_wrong_stops'] = outcome['wrong_stops']
        if result['sound']:
            assert outcome['wrong_stops'] == 0
        elif outcome['wrong_stops']:
            job = mapping['actual']
            if job not in witness_cache:
                path = RAW/job/'moves-262144/boards.npz'
                assert sha(path) == summary['input_sha256'][str(path.relative_to(ROOT))]
                with np.load(path, allow_pickle=False) as data:
                    bad = np.flatnonzero(data['stopped'] & ~data['valid'])
                    k = int(bad[0])
                    witness_cache[job] = dict(job=job, local_index=k, puzzle_index=int(data['puzzle'][k]),
                        board=data['boards'][k].astype(int).tolist(), solution=data['solutions'][k].astype(int).tolist(),
                        free=data['free'][k].astype(bool).tolist(), source_archive=str(path.relative_to(ROOT)), source_sha256=sha(path))
            result['witness_job'] = job
        rows.append(result)
    groups = []
    for arm in ['adamw', 'paper_pl', 'long_pl', 'restricted', 'chernoff9']:
        selected = [r for r in rows if r['arm'] == arm]
        groups.append(dict(arm=arm, sound=sum(r['sound'] for r in selected),
                           unsound_witness=sum('witness_job' in r for r in selected),
                           unresolved=sum(not r['sound'] and 'witness_job' not in r for r in selected)))
    result = dict(status='produced_pending_independent_verification', utc=datetime.now(timezone.utc).isoformat(),
        scope='Post-hoc exact graph diagnostic using supplied Sudoku geometry; no new fits, searches, inputs, or hypothesis test.',
        parent_sha256=sha(PARENT/'summary.json'), supports_sha256=sha(PUBLIC/'supports.npz'),
        producer_sha256=sha(__file__), rows=rows, witnesses=list(witness_cache.values()), grouped=groups,
        seconds=time.perf_counter()-start)
    (PUBLIC/'certificates.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(grouped=groups, unique_witnesses=len(witness_cache), seconds=result['seconds'])))


if __name__ == '__main__':
    main()
