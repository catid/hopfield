"""Bounded CPU acquisition/replay pipeline for source-only H1ai."""
from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
import sys
from run import HERE


def one(seed):
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for script in ('confidence_budget_acquire.py', 'audit_confidence_budget_acquisition.py'):
        subprocess.run([sys.executable, str(HERE/script), '--seed', str(seed)], check=True, env=env)


def main():
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(one, range(32)))
    subprocess.run([sys.executable, str(HERE/'summarize_confidence_budget_source.py')], check=True)


if __name__ == '__main__':
    main()
