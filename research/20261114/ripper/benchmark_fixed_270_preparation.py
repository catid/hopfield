"""Bounded synthetic startup/batching benchmark; no study IDs or LP outcomes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse, hashlib, json, os, resource, statistics, subprocess, sys, time
import numpy as np
from fixed_270_confirm_study import HERE, ROOT, make_masks, observe, sha, save


def toy_jobs(kind, indices):
    output = {}
    for index in indices:
        mask = make_masks(984000+index)
        digest = hashlib.sha256(mask.tobytes())
        if kind == 'counts':
            rng = np.random.default_rng(984100+index)
            clean = rng.integers(0, 9, (270, 81), dtype=np.int8)
            noise = dict(iid_uniforms=rng.random((270, 81)), iid_offsets=rng.integers(1, 9, (270, 81), dtype=np.int8))
            for n in [180, 270]:
                values = observe(mask[:n], clean[:n], {k: v[:n] for k, v in noise.items()})
                for key in ['values', 'joint_count', 'equal_count']:
                    digest.update(values[key].tobytes())
        output[str(index)] = digest.hexdigest()
    return output


def benchmark():
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    def invoke(kind, indices):
        completed = subprocess.run([sys.executable, str(Path(__file__)), '--kind', kind, '--indices', ','.join(map(str, indices))],
                                   cwd=ROOT, env=env, check=True, capture_output=True, text=True)
        return json.loads(completed.stdout)
    results = []
    for kind, modes in [('masks', ['fresh', 'batch']), ('counts', ['fresh', 'persistent'])]:
        reference = None
        for mode in modes:
            samples = []
            for repetition in range(3):
                before = resource.getrusage(resource.RUSAGE_CHILDREN)
                started = time.perf_counter()
                groups = [[i] for i in range(64)] if mode == 'fresh' else [list(range(64))] if mode == 'batch' else [list(range(i, 64, 8)) for i in range(8)]
                with ThreadPoolExecutor(max_workers=8) as pool:
                    parts = list(pool.map(lambda indices: invoke(kind, indices), groups))
                actual = {}
                for part in parts:
                    assert not actual.keys() & part.keys()
                    actual.update(part)
                assert sorted(map(int, actual)) == list(range(64))
                if reference is None:
                    reference = actual
                else:
                    assert actual == reference
                elapsed = time.perf_counter()-started
                after = resource.getrusage(resource.RUSAGE_CHILDREN)
                samples.append(dict(wall_seconds=elapsed, child_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime))
            results.append(dict(kind=kind, mode=mode, child_processes=len(groups), samples=samples,
                                median_wall_seconds=statistics.median(v['wall_seconds'] for v in samples),
                                median_child_cpu_seconds=statistics.median(v['child_cpu_seconds'] for v in samples),
                                outputs_sha256=hashlib.sha256(json.dumps(reference, sort_keys=True).encode()).hexdigest()))
    result = dict(status='passed', scope='64 unrelated synthetic mask/count bundles per repetition, three repetitions per mode; no source IDs, study masks or LP outcomes. Startup benchmark is not full scientific execution.',
                  workers=8, all_outputs_identical_within_kind=True, results=results, script_sha256=sha(__file__))
    save(HERE/'fixed-270-preparation-benchmark.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['masks', 'counts'])
    parser.add_argument('--indices')
    args = parser.parse_args()
    if args.kind:
        print(json.dumps(toy_jobs(args.kind, list(map(int, args.indices.split(','))))))
    else:
        benchmark()
