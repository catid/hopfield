"""Prepare all six shared data-efficiency splits from the seeded recreation.

The original unseeded reference was not released. These are explicitly a new
protocol reproduction. Seed workers write disjoint directories and use the
unmodified pinned builders. GPU work can overlap this CPU-only preparation.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reproduction/baselines"
DATA = BASE / "experiments/runs/trm/data"
OUT = ROOT / "artifacts/data_efficiency"
SMALL = [1, 3, 10, 12, 14, 17, 19, 21, 23, 25, 27, 30, 50, 70, 100, 150, 200, 300, 500, 700, 3000, 10000]
CRAM = [1, 3, 10, 12, 14, 17, 19, 21, 23, 25, 27, 30, 100, 300, 3000, 10000]
ALL = [1, 3, 10, 12, 14, 17, 19, 21, 23, 25, 27, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 3000, 10000, 30000, 100000]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write(path, record):
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(path)


def validate_preparation(record, seed, baseline_data=False):
    """A completed preparation marker must still name intact arrays and reference."""
    assert record['reference_provenance_sha256'] == sha(DATA / 'sudoku-extreme-1k-aug-1000/build_provenance.json')
    families = [('sudoku-smalldata', SMALL if baseline_data else CRAM)]
    if baseline_data:
        families.append(('sudoku-smalldata-v99', ALL))
    for family, sizes in families:
        for size in sizes:
            marker = DATA / family / f's{seed}-n{size}/subset.json'
            relative = str(marker.relative_to(DATA))
            assert record['subset_manifest_sha256'][relative] == sha(marker)
            manifest = json.loads(marker.read_text())
            assert manifest['seed'] == seed and manifest['size'] == size
            for name, expected in manifest['sha256'].items():
                assert sha(marker.parent / 'train' / f'all__{name}.npy') == expected, f'Changed dataset array: {marker.parent}/{name}'


def worker(seed, baseline_data=False):
    # A second preparation process cannot concurrently write the same builder
    # temporary files or erase the first process's phase history.
    with (OUT / f'preparation-seed{seed}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _worker(seed, baseline_data)


def _worker(seed, baseline_data=False):
    path = OUT / f"preparation-seed{seed}.json"
    previous = None
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("completed_utc") and (not baseline_data or previous.get('scope') == 'CRAM and baseline datasets'):
            validate_preparation(previous, seed, baseline_data)
            if not baseline_data:
                previous['scope'] = 'CRAM only; user-confirmed scope'
                write(path, previous)
            return previous
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               TRM_DATA=str(DATA), TRM_SRC=str(ROOT / "upstream/third_party/trm"),
               SUDOKU_EXTREME_CSV=str(ROOT / "upstream/data/sudoku-extreme/train.csv"))
    record = previous or {"seed": seed, "started_utc": now(), "phases": [],
              "protocol": "New shared protocol reproduction, using explicitly seeded reference42; original unseeded reference unavailable.",
              "reference_provenance_sha256": sha(DATA / "sudoku-extreme-1k-aug-1000/build_provenance.json")}
    record['scope'] = 'CRAM and baseline datasets' if baseline_data else 'CRAM only; user-confirmed scope'
    phases = [("trm_smalldata.py", SMALL if baseline_data else CRAM)]
    if baseline_data:
        phases.append(("trm_smalldata_v99.py", ALL))
    for script, sizes in phases:
        src = ROOT / "upstream/baselines/trm" / script
        family = 'sudoku-smalldata' if script == 'trm_smalldata.py' else 'sudoku-smalldata-v99'
        prior = [phase for phase in record['phases'] if phase['builder_sha256'] == sha(src) and phase['exit_code'] == 0]
        if prior and all((DATA / family / f's{seed}-n{k}/subset.json').exists() for k in sizes):
            continue
        argv = [str(BASE / ".venv/bin/python"), "-u", str(src), "build", "--seeds", str(seed),
                "--sizes", *map(str, sizes)]
        phase = {"argv": argv, "builder_sha256": sha(src), "started_utc": now()}
        begin = time.perf_counter()
        with (OUT / f"seed{seed}-{script}.log").open("a") as log:
            result = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        phase.update(exit_code=result.returncode, elapsed_s=time.perf_counter() - begin, completed_utc=now())
        record["phases"].append(phase)
        write(path, record)
        result.check_returncode()
        print(json.dumps({"seed": seed, "builder": script, "elapsed_s": phase["elapsed_s"]}), flush=True)
    families = ("sudoku-smalldata", "sudoku-smalldata-v99") if baseline_data else ("sudoku-smalldata",)
    subsets = [p for family in families
               for p in sorted((DATA / family).glob(f"s{seed}-n*/subset.json"))]
    record["subset_manifest_sha256"] = {str(p.relative_to(DATA)): sha(p) for p in subsets}
    validate_preparation(record, seed, baseline_data)
    record["completed_utc"] = now()
    write(path, record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-baseline-data', action='store_true', help='Optional expanded preparation, outside the confirmed CRAM-only study')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    assert (DATA / "sudoku-extreme-1k-aug-1000/build_provenance.json").exists()
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = [future.result() for future in [pool.submit(worker, seed, args.include_baseline_data) for seed in range(6)]]
    write(OUT / "preparation.json", results)
    print("All six seeded shared datasets prepared and validated for the declared scope.", flush=True)
