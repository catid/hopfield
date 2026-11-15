"""Bounded subprocess probes for initial hashing and descendant budget charging."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from common import SRC, save, sha

WORKER = '''import json,os,resource,signal,subprocess,sys,time
from pathlib import Path
out=Path(__file__).parent/'out';start=time.perf_counter();child=None
mode=Path(__file__).with_name('mode.txt').read_text()
def stop(*args):
 if child is not None:child.terminate();child.wait()
 raise RuntimeError('synthetic monitor stop')
signal.signal(signal.SIGUSR1,stop)
try:
 if mode=='descendants':
  child=subprocess.Popen([sys.executable,'-c','import time; s=time.process_time(); exec("while time.process_time()-s<0.06: pass")'])
  s=time.process_time()
  while time.process_time()-s<0.03:pass
  child.wait();child=None
 elif mode=='cpu_cap':
  while True:pass
 elif mode=='wall_cap':time.sleep(10)
 r=resource.getrusage(resource.RUSAGE_SELF);c=resource.getrusage(resource.RUSAGE_CHILDREN)
 (out/'study-result.json').write_text(json.dumps(dict(status='complete_verified',stage_wall_seconds=time.perf_counter()-start,stage_cpu_seconds=r.ru_utime+r.ru_stime+c.ru_utime+c.ru_stime)))
except BaseException:
 (out/'study-result.json').write_text(json.dumps(dict(status='failed_or_incomplete',stage_wall_seconds=time.perf_counter()-start)))
 raise
'''


def probe(folder, mode):
    import gate
    folder.mkdir(parents=True, exist_ok=False)
    (folder/'study.py').write_text(WORKER)
    (folder/'protocol.json').write_text('{}\n')
    (folder/'mode.txt').write_text(mode)
    gate.SRC, gate.OUT = folder, folder/'out'
    def frozen():
        start = time.process_time()
        while time.process_time()-start < (0.12 if mode == 'initial_hash' else 0.04):
            pass
    gate.frozen = frozen
    limits = dict(initial_hash=(0.08, 3), descendants=(1, 3), cpu_cap=(0.3, 3), wall_cap=(1, 0.3))
    try:
        gate.main(*limits[mode])
    except AssertionError:
        pass
    result = json.loads((folder/'out/capacity-gate.json').read_text())
    assert result['full_monitor_process_lifetime_CPU_charged']
    assert result['initial_source_hashing_inside_wall_and_CPU_budget']
    if mode == 'initial_hash':
        assert result['monitor_expired'] and result['aggregate_cpu_seconds'] >= 0.12
        assert not (folder/'out/study-result.json').exists()
    elif mode == 'descendants':
        assert result['status'] == 'complete_verified' and result['aggregate_cpu_seconds'] >= 0.13
        assert result['stage_cpu_seconds'] >= 0.09
    else:
        assert result['status'] == 'failed_or_incomplete' and result['monitor_expired']
    save(folder/'probe-receipt.json', result)


def run(folder):
    folder.mkdir(parents=True, exist_ok=False)
    modes = ['initial_hash', 'descendants', 'cpu_cap', 'wall_cap']
    rows = []
    for mode in modes:
        completed = subprocess.run([sys.executable, str(SRC/'monitor_gate.py'), '--probe',
                                    mode, str(folder/mode)], capture_output=True, text=True)
        (folder/(mode+'.stdout.txt')).write_text(completed.stdout)
        (folder/(mode+'.stderr.txt')).write_text(completed.stderr)
        assert completed.returncode == 0, completed.stderr
        rows.append(dict(mode=mode, receipt=json.loads((folder/mode/'probe-receipt.json').read_text())))
    result = dict(status='passed_monitor_accounting_probes', probes=rows,
                  gate_source_sha256=sha(SRC/'gate.py'), probe_source_sha256=sha(SRC/'monitor_gate.py'),
                  fresh_truth_draws=0, fresh_source_draws=0, actual_oracle_calls=0)
    save(folder/'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', choices=['initial_hash', 'descendants', 'cpu_cap', 'wall_cap'])
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    if args.probe:
        probe(args.folder, args.probe)
    else:
        print(run(args.folder)['status'])
