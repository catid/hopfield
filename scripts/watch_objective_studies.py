#!/usr/bin/env python3
"""Durable collection, guarded queue refills and final objective-study export."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import objective_queue
import d2_queue
import figure4_queue
from export_objective_studies import export
from migrate_pending_figure4 import auto_once
from objective_queue import ROOT, now, write


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--once', action='store_true')
    p.add_argument('--interval', type=float, default=60); args = p.parse_args()
    root = ROOT / 'artifacts/objective_studies'; root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'supervisor.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        started = time.monotonic(); errors = []
        write(root / 'supervisor.json', {'pid': os.getpid(), 'status': 'collecting', 'utc': now()})
        tasks = [(objective_queue.collect, 'objective_jobs.json'), (d2_queue.collect, 'd2_supplemental_jobs.json'),
                 (figure4_queue.collect, 'figure4_jobs.json')]
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [(name, pool.submit(fn, SimpleNamespace(config=str(ROOT / 'configs' / name)))) for fn, name in tasks]
            for name, future in futures:
                try: future.result()
                except Exception as exc: errors.append({'phase': name, 'error': repr(exc), 'traceback': traceback.format_exc()})
        dispatch = None
        if not errors:
            try:
                dispatch = auto_once(max_jobs=4)
                write(root / 'dispatcher.json', dispatch)
            except Exception as exc: errors.append({'phase': 'dispatch', 'error': repr(exc), 'traceback': traceback.format_exc()})
        summary = None
        if not errors:
            try: summary = export()
            except Exception as exc: errors.append({'phase': 'export', 'error': repr(exc), 'traceback': traceback.format_exc()})
        status = 'complete' if summary and summary['status'] == 'complete' and not errors else 'incomplete'
        write(root / 'supervisor.json', {'pid': os.getpid(), 'status': status, 'utc': now(), 'errors': errors,
              'cycle_wall_seconds': time.monotonic()-started, 'next_collection_seconds': args.interval,
              'dispatch_status': dispatch.get('status') if dispatch else None,
              'scope': '84 Table 7 jobs, 9 D2 development-only jobs, 100 Figure 4 curves. Guarded dispatcher may refill released authorized lanes with untouched deferred Figure 4 jobs.'})
        if args.once or status == 'complete': return
        time.sleep(min(60, max(5, args.interval)))


if __name__ == '__main__': main()
