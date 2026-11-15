"""Prospective source-only numerical gate for tenfold weaker PL regularization.

Reuses the hash-pinned H4s matched-size numerical comparison unchanged, with
new output/protocol paths and lambda=1/30000. No fresh source or target access.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'artifacts/research/20261114/foureyes/weak_penalty_gate'
PROTOCOL = HERE/'weak_penalty_gate.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists()
    parent = HERE/'noisy_pl_long.json';p = json.loads(parent.read_text())
    p.update(id='H4v-source-only-weaker-penalty-numerical-gate',frozen_utc=datetime.now(timezone.utc).isoformat(),
        hypothesis='Tenfoldweakerpenalty retains matching selectedsupports andcoefficientseparation acrossGPU32/GPU64/CPU32 onallfourpriorN600noisyfixtures.',
        scope='Source-onlynumericalgate; no newscientificfits/testqueries/inference. Freshscientificprotocol will be frozen separately if gatepasses.',
        rationale='Exactuniform-digit objectivecertificates madezero theuniquenominalpenaltyminimum inall64priornoisybanks. Lambda/10 isbelowalltheir maximumorigin gradients; this motivates a newpenalty test, not a convergenceclaim.',
        original_protocol_sha256=sha(parent),max_gate_queue_seconds=120)
    p['long_pl']['lam'] = 1/30000
    p['source_sha256'][str(Path(__file__).relative_to(ROOT))] = sha(__file__)
    p['gate'] = 'AllfourpriorN600fixtures mustpass unchanged coefficient-error<quarterthresholdseparation and exactsupportidentity. Preserveallfailures; no tuning or replacement.'
    p['matched_gate']['criterion'] = p['gate']
    p['matched_gate']['banks'] = [1,2,3,4]
    p.pop('gate_amendment',None)
    PROTOCOL.write_text(json.dumps(p,indent=2)+'\n');OUT.mkdir(parents=True)
    print(json.dumps(dict(protocol_sha256=sha(PROTOCOL),lambda_value=p['long_pl']['lam'])))


def run_bank(bank):
    gpu = bank-1
    lock = (ROOT/f'artifacts/gpu{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    uuid = subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    import gate_long_matched as gate
    gate.OUT = OUT;gate.PROTOCOL = PROTOCOL
    sys.argv = [str(__file__),'--bank',str(bank)]
    gate.main()


def launch():
    start = time.monotonic();workers = [];logs = [];samples = []
    try:
        for bank in [1,2,3,4]:
            log = (OUT/f'bank{bank}.log').open('x');logs.append(log)
            workers.append(subprocess.Popen([str(ROOT/'upstream/.venv/bin/python'),str(Path(__file__)),'bank','--bank',str(bank)],
                stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True))
        while any(w.poll() is None for w in workers):
            assert time.monotonic()-start<120
            samples.append(dict(elapsed=time.monotonic()-start,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True)))
            time.sleep(2)
        receipts = [json.loads((OUT/f'matched-gate-b{b}.json').read_text()) for b in [1,2,3,4]]
        passed = all(w.returncode==0 for w in workers) and all(r['status']=='passed' for r in receipts)
        result = dict(status='passed' if passed else 'failed',protocol_sha256=sha(PROTOCOL),receipts=receipts,
            wall_seconds=time.monotonic()-start,returncodes=[w.returncode for w in workers],samples=samples)
        (OUT/'gate.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ['samples','receipts']}))
    finally:
        for w in workers:
            if w.poll() is None:
                os.killpg(w.pid,signal.SIGTERM)
                try:w.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(w.pid,signal.SIGKILL);w.wait()
        for log in logs:log.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','launch','bank']);parser.add_argument('--bank',type=int)
    args=parser.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='launch':launch()
    else:run_bank(args.bank)
