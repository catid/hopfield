"""Complete-source, D4 minimum-rule identification; no target data access."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import datetime as dt
import hashlib
import itertools
import json
import math
from pathlib import Path
import signal
import socket
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / 'research/20261114/loki/h3x'
OUT = ROOT / 'artifacts/research/20261114/loki/h3x'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p, value):
    Path(p).write_text(json.dumps(value, indent=2) + '\n')

def main():
    assert socket.gethostname() == 'loki'
    assert not (OUT / 'source-result.json').exists()
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    cpu_start = time.process_time()
    def expired(*unused):
        raise TimeoutError('Frozen 60-second source selection and certificate budget reached')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 60)
    try:
        protocol = json.loads((SRC / 'protocol.json').read_text())
        for group in ['source_sha256', 'source_data_sha256']:
            for path, digest in protocol[group].items():
                assert sha(ROOT / path) == digest, path
        source_path = ROOT / protocol['source']['path']
        positives = np.load(source_path)
        n = 12
        assert positives.shape == (176, n*n) and positives.dtype == bool
        assert np.all(positives.reshape(-1, n, n).sum(2) == 1)
        columns = positives.reshape(-1, n, n).argmax(2).astype(np.uint8)
        positive_keys = {tuple(map(int, row)) for row in columns}
        assert len(positive_keys) == len(positives)
        counts = {(dr, dc): 0 for dr in range(n) for dc in range(1-n, n) if dr > 0 or dc > 0}
        for board in positives:
            for x, y in itertools.combinations(np.flatnonzero(board), 2):
                counts[int(y//n-x//n), int(y%n-x%n)] += 1
        lines = []
        candidates = []
        for a, b in itertools.product(range(-2, 3), repeat=2):
            if math.gcd(abs(a), abs(b)) != 1 or a < 0 or (a == 0 and b < 0):
                continue
            members = [v for (dr, dc), v in counts.items() if a*dr+b*dc == 0]
            retained = members.count(0) >= 3 and not any(members)
            candidates.append(dict(normal=[a, b], zero_offsets=members.count(0), observed_offsets=sum(v > 0 for v in members), retained=retained))
            if retained:
                lines.append((a, b))
        assert (0, 1) in lines and (1, 0) in lines
        # A row exclusion and exactly n objects imply precisely one object per row.
        # Distinct line-coordinate values forbid each inferred line attack.
        universe = []
        occupied = [set() for _ in lines]
        placed = []
        nodes = 0
        def visit(row):
            nonlocal nodes
            nodes += 1
            if row == n:
                universe.append(tuple(placed))
                return
            for col in range(n):
                values = [a*row+b*col for a, b in lines]
                if any(value in used for value, used in zip(values, occupied)):
                    continue
                for value, used in zip(values, occupied):
                    used.add(value)
                placed.append(col)
                visit(row+1)
                placed.pop()
                for value, used in zip(values, occupied):
                    used.remove(value)
        visit(0)
        universe = np.asarray(universe, np.uint8)
        assert len(universe) == len({tuple(map(int, row)) for row in universe})
        is_positive = np.array([tuple(map(int, row)) in positive_keys for row in universe])
        assert int(is_positive.sum()) == len(positives)
        zeros = {o for o, v in counts.items() if v == 0 and not any(a*o[0]+b*o[1] == 0 for a, b in lines)}
        def canonical(dr, dc):
            return (-dr, -dc) if dr < 0 or (dr == 0 and dc < 0) else (dr, dc)
        generators = sorted({tuple(sorted(map(abs, o))) for o in zeros})
        orbits = []
        for small, large in generators:
            orbit = sorted({canonical(sign1*x, sign2*y) for x, y in [(small, large), (large, small)] for sign1 in [-1, 1] for sign2 in [-1, 1]})
            assert set(orbit) <= zeros
            orbits.append(orbit)
        assert set().union(*(set(o) for o in orbits)) == zeros
        incidence = np.zeros((len(universe), len(orbits)), bool)
        for k, orbit in enumerate(orbits):
            for dr, dc in orbit:
                incidence[:, k] |= (universe[:, dr:].astype(np.int16)-universe[:, :n-dr].astype(np.int16) == dc).any(1)
        assert not incidence[is_positive].any()
        negative = ~is_positive
        subsets = []
        for mask in range(1 << len(orbits)):
            chosen = [k for k in range(len(orbits)) if mask & (1 << k)]
            rejected = incidence[:, chosen].any(1)
            uncovered = np.flatnonzero(negative & ~rejected)
            subsets.append(dict(mask=mask, cardinality=len(chosen), orbit_indices=chosen,
                                rejected_negatives=int((negative & rejected).sum()), covers_all_negatives=len(uncovered) == 0,
                                first_uncovered_universe_index=int(uncovered[0]) if len(uncovered) else None,
                                first_uncovered_columns=universe[uncovered[0]].astype(int).tolist() if len(uncovered) else None))
        covers = [s for s in subsets if s['covers_all_negatives']]
        minimum = min((s['cardinality'] for s in covers), default=None)
        minima = [s['mask'] for s in covers if s['cardinality'] == minimum]
        np.savez_compressed(OUT / 'source-certificate.npz', source_columns=columns, universe_columns=universe,
                            is_positive=is_positive, incidence=incidence, lines=np.asarray(lines, np.int16), generators=np.asarray(generators, np.int16))
        result = dict(status='selected_pending_independent_certificate', utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                      source_n=n, source_positives=len(positives), source_universe_boards=len(universe), source_negatives=int(negative.sum()),
                      source_completeness_supplied=True, source_zero_offsets=sum(v == 0 for v in counts.values()),
                      line_candidates=candidates, inferred_lines=lines, off_line_zero_offsets=len(zeros),
                      orbits=[dict(index=k, generator=g, offsets=o, source_negative_coverage=int(incidence[negative, k].sum())) for k, (g, o) in enumerate(zip(generators, orbits))],
                      subsets=subsets, minimum_cardinality=minimum, all_minimum_masks=minima, unique_minimum=len(minima) == 1,
                      selected_generators=[[generators[k] for k in range(len(orbits)) if mask & (1 << k)] for mask in minima],
                      enumeration_nodes=nodes, selection_seconds=time.perf_counter()-start,
                      protocol_sha256=sha(SRC / 'protocol.json'), source_data_sha256=sha(source_path),
                      certificate_npz_sha256=sha(OUT / 'source-certificate.npz'), no_target_file_opened=True)
        save(OUT / 'source-selection.json', result)
        import verify_source
        certificate = verify_source.verify(result, OUT / 'source-certificate.npz', source_path)
        elapsed, cpu = time.perf_counter()-start, time.process_time()-cpu_start
        assert elapsed < 60 and cpu < 60
        signal.setitimer(signal.ITIMER_REAL, 0)
        save(OUT / 'source-verification.json', certificate)
        result.update(status='source_selected_and_independently_certified', total_source_wall_seconds=elapsed, total_source_cpu_seconds=cpu,
                      verification_sha256=sha(OUT / 'source-verification.json'), selection_sha256=sha(OUT / 'source-selection.json'),
                      selected_before_any_target_diagnostic=True)
        save(OUT / 'source-result.json', result)
        print(json.dumps({k: result[k] for k in ['status', 'source_universe_boards', 'source_negatives', 'inferred_lines', 'orbits', 'minimum_cardinality', 'all_minimum_masks', 'selected_generators', 'total_source_wall_seconds', 'total_source_cpu_seconds']}), flush=True)
    except BaseException as exc:
        save(OUT / 'failure.json', dict(status='failed', error=repr(exc), seconds=time.perf_counter()-start))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

if __name__ == '__main__':
    main()
