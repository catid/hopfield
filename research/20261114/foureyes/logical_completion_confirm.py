"""H4u: fresh 40k-input confirmation of H4t's unchanged one-point target."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/research/20261114/foureyes/logical_completion_confirm'
PUBLIC = ROOT/'results/research/foureyes/logical_completion_confirm'
PROTOCOL = HERE/'logical_completion_confirm.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def freeze():
    from verify_sudoku_soundness import verify
    assert verify()['status'] == 'passed'
    assert not OUT.exists() and not PROTOCOL.exists()
    parent = ROOT/'artifacts/research/20261114/foureyes/logical_completion'
    old = json.loads((parent/'prepared.json').read_text())
    p = json.loads((HERE/'logical_completion.json').read_text())
    p.update(id='H4u-logical-completion-confirmation', frozen_utc=datetime.now(timezone.utc).isoformat(),
        rows_half_open=[340000,380000],
        hypothesis='Fresh40k-input confirmation of unchanged >=1pp gain from logically implied edges in the same13fixedAdamW graphs.',
        selection=[r for r in p['selection'] if r['source_arm'] == 'adamw'],
        secondary='No additional source-model arms. H4t remains inconclusive; no pooling or retrospective replacement.',
        input_scope='New40k contiguous slice of reused officialcorpus;13fixedhistoricalmodels andsearchseed0. Not a newtraining-bank replication.',
        adaptive_reason='H4t gain0.721538pp with95CI[0.209981,1.230019] was inconclusive for unchanged1pp target. One larger independentcohort fixed before outcomes.',
        budget=dict(max_queue_wall_seconds=300, max_gpu_minutes=20, gpus=[0,1,2,3]))
    p['primary']['interval'] = 'two-sided95percent paired input-cluster bootstrap; average13fixedmodels inside each of40000inputs;5000draws;seed20261115666'
    p['source_sha256'][str(Path(__file__).relative_to(ROOT))] = sha(__file__)
    p['frozen_inputs'][str((parent/'prepared.json').relative_to(ROOT))] = sha(parent/'prepared.json')
    p['frozen_inputs']['results/research/foureyes/logical_completion/summary.json'] = sha(ROOT/'results/research/foureyes/logical_completion/summary.json')
    save(PROTOCOL, p)
    mappings = [r for r in old['mappings'] if r['source_arm'] == 'adamw']
    jobs = []
    for job in old['jobs']:
        if job['job_id'] not in {r['actual'] for r in mappings}:continue
        source = ROOT/job['model_path'];assert sha(source) == job['model_sha256']
        dest = OUT/'models'/source.name;dest.parent.mkdir(parents=True, exist_ok=True);shutil.copyfile(source, dest)
        jobs.append(dict(job, model_path=str(dest.relative_to(ROOT)), model_sha256=sha(dest), status='pending'))
    assert len(jobs) == 18 and len(mappings) == 26
    save(OUT/'prepared.json', dict(protocol_sha256=sha(PROTOCOL), jobs=jobs, mappings=mappings, logical_cells=26, unique_searches=18))
    print(json.dumps(dict(protocol_sha256=sha(PROTOCOL), unique_searches=18, logical_cells=26)))


def worker(gpu, initialize=False):
    import sys
    import noisy_pl_long_search as search
    search.OUT = OUT;search.PROTOCOL = PROTOCOL
    sys.argv = [str(__file__), '--initialize'] if initialize else [str(__file__), '--gpu', str(gpu)]
    search.main()


def launch():
    worker(None, initialize=True)
    start = time.monotonic();workers = [];handles = [];samples = []
    try:
        for gpu in range(4):
            handle = (OUT/f'gpu{gpu}.log').open('x');handles.append(handle)
            workers.append(subprocess.Popen([str(ROOT/'upstream/.venv/bin/python'), str(Path(__file__)), 'worker', '--gpu', str(gpu)],
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True, cwd=ROOT))
        while any(w.poll() is None for w in workers):
            elapsed = time.monotonic()-start
            assert elapsed < 300 and all(w.poll() in [None,0] for w in workers)
            q = json.loads((OUT/'queue.json').read_text())
            samples.append(dict(elapsed=elapsed, complete=sum(r['status']=='complete' for r in q), running=sum(r['status']=='running' for r in q),
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True)))
            time.sleep(2)
        assert all(w.returncode == 0 for w in workers)
        result = dict(status='complete', wall_seconds=time.monotonic()-start, samples=samples, source_sha256=sha(__file__))
        save(OUT/'launch-result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='samples'}))
    finally:
        for w in workers:
            if w.poll() is None:
                os.killpg(w.pid,signal.SIGTERM)
                try:w.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(w.pid,signal.SIGKILL);w.wait()
        for h in handles:h.close()


def audit_results():
    import audit_noisy_pl_long as auditor
    auditor.OUT = OUT
    p = json.loads(PROTOCOL.read_text());prepared = json.loads((OUT/'prepared.json').read_text())
    assert prepared['protocol_sha256'] == sha(PROTOCOL)
    for path,digest in (p['source_sha256'] | p['frozen_inputs']).items():assert sha(ROOT/path) == digest
    q = json.loads((OUT/'queue.json').read_text());assert all(r['status']=='complete' for r in q)
    with np.load(ROOT/'upstream/data/full-test/full-all.npz',allow_pickle=False) as z:
        lo,hi = p['rows_half_open']
        np.savez_compressed(OUT/'audit-target.npz', solutions=z['solutions'][lo:hi].astype(np.int8)-1,free=~z['clue_mask'][lo:hi].astype(bool),puzzle=np.arange(lo,hi))
    with ProcessPoolExecutor(max_workers=4) as pool:results = list(pool.map(auditor.audit_job,q))
    lookup = {r['job_id']:r for r in results}
    mapping = {(r['bank'],r['arm']):r['actual'] for r in prepared['mappings']}
    banks = sorted({b for b,a in mapping})
    before = np.stack([lookup[mapping[b,'original']]['good'] for b in banks]).astype(np.int8)
    after = np.stack([lookup[mapping[b,'completed']]['good'] for b in banks]).astype(np.int8)
    delta = (after-before).mean(0);rng = np.random.default_rng(20261115666)
    draws = np.array([delta[rng.integers(0,40000,40000)].mean()*100 for _ in range(5000)])
    interval = np.quantile(draws,[.025,.975]).tolist()
    wrong = sum(int((r['stopped'] & ~r['good']).sum()) for r in results)
    decision = 'falsified' if wrong or interval[1]<1 else 'supported' if interval[0]>=1 else 'inconclusive'
    PUBLIC.mkdir(parents=True,exist_ok=True);ids = sorted(lookup)
    np.savez_compressed(PUBLIC/'audited-outcome-bits.npz',job_ids=np.array(ids),valid=np.packbits(np.stack([lookup[k]['good'] for k in ids]),axis=1),stopped=np.packbits(np.stack([lookup[k]['stopped'] for k in ids]),axis=1))
    evidence = {}
    for r in results:evidence.update(r['input_sha256'])
    result = dict(status='validated', decision=decision, gain_pp=float(delta.mean()*100), paired_input95_interval_pp=interval,
        original_accuracy_pct=float(before.mean()*100), completed_accuracy_pct=float(after.mean()*100), unique_wrong_stops=wrong,
        fixed_models=13,unique_inputs=40000,logical_cells=26,unique_searches=18,boards_audited=720000,
        model_mappings=prepared['mappings'],protocol_sha256=sha(PROTOCOL),input_sha256=evidence,
        outcome_bits_sha256=sha(PUBLIC/'audited-outcome-bits.npz'),scope=p['input_scope'],launch=json.loads((OUT/'launch-result.json').read_text()))
    save(PUBLIC/'summary.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['input_sha256','model_mappings','launch']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','launch','worker','audit']);parser.add_argument('--gpu',type=int)
    args = parser.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='launch':launch()
    elif args.action=='worker':worker(args.gpu)
    else:audit_results()
