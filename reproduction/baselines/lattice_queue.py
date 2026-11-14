"""Exclusive ripper GPU0 queue: parity, smoke, three frozen fits, tests, warm timing."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,fcntl,json,math,os,subprocess,time
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[1]
ART=ROOT/'artifacts';ART.mkdir(parents=True,exist_ok=True)
PY=ROOT/'.venv/bin/python';CODE=ROOT/'code';RUNS=CODE/'runs/trmp-lattice'


def event(**kw):
    kw['utc']=datetime.now(timezone.utc).isoformat()
    with (ART/'lattice-queue.jsonl').open('a') as f:f.write(json.dumps(kw)+'\n')
    print(json.dumps(kw),flush=True)


def stage(name,argv,cache):
    env=dict(os.environ,TORCHINDUCTOR_CACHE_DIR=str(cache))
    event(stage=name,status='started',argv=[str(a) for a in argv])
    begin=time.monotonic()
    with (ART/f'{name}.log').open('a') as log:
        p=subprocess.Popen([str(a) for a in argv],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT)
        event(stage=name,status='running',pid=p.pid)
        rc=p.wait()
    event(stage=name,status='completed' if rc==0 else 'failed',exit_code=rc,wall_seconds=time.monotonic()-begin)
    if rc:raise SystemExit(f'{name} failed: {rc}')


def active(gpu):
    uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()
    return [r for r in rows if r.split(',')[0].strip()==uuid]


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=0);p.add_argument('--wait-pid',type=int);p.add_argument('--warm',action='store_true');a=p.parse_args()
    if a.wait_pid:
        while Path(f'/proc/{a.wait_pid}').exists():
            cmd=Path(f'/proc/{a.wait_pid}/cmdline').read_bytes().replace(b'\0',b' ')
            if b'run_hand_sudoku.py' not in cmd:raise SystemExit('PID reused; stop for coordinator review')
            event(status='waiting-for-hand-sudoku',pid=a.wait_pid);time.sleep(15)
    lock=(WORK/'artifacts'/f'gpu{a.gpu}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    holders=active(a.gpu)
    if holders:raise SystemExit(f'GPU still occupied after prior queue exited: {holders}')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),V1_DATA=str(WORK/'upstream/data'),CRAM_DATA=str(WORK/'upstream/data'),
                      WANDB_MODE='disabled',WANDB_DISABLED='true',OMP_NUM_THREADS='8')
    event(status='exclusive-gpu-acquired',gpu=a.gpu,lock=str(WORK/'artifacts'/f'gpu{a.gpu}.lock'))
    if not (ROOT/'smoke/lattice-parity/check.json').exists():
        stage('lattice-parity',[PY,ROOT/'check_lattice_gpu.py'],ROOT/'smoke/lattice-parity/.inductor')
    assert json.loads((ROOT/'smoke/lattice-parity/check.json').read_text())['passed']
    if not (ROOT/'smoke/lattice/fit.json').exists():
        stage('lattice-smoke',[PY,CODE/'trmp/lattice_fit.py','--seed','0','--steps','32','--schedule-steps','20000','--no-validate','--log-every','8','--out',ROOT/'smoke/lattice'],ROOT/'smoke/lattice/.inductor')
    logs=[json.loads(x) for x in (ROOT/'smoke/lattice/log.jsonl').read_text().splitlines()]
    assert logs[-1]['step']==32 and all(math.isfinite(r[k]) for r in logs for k in ['loss','gradient_norm'])
    event(status='smoke-validated',final_loss=logs[-1]['loss'],warm_s_per_update=logs[-1]['s_per_update'],peak_mem_gb=max(r['peak_mem_gb'] for r in logs))
    for seed in range(3):
        out=RUNS/f'sudoku-extreme-s{seed}'
        if not (out/'fit.json').exists():stage(f'lattice-seed{seed}-fit',[PY,CODE/'trmp/lattice_fit.py','--seed',seed,'--out',out],out/'.inductor')
        fit=json.loads((out/'fit.json').read_text());assert fit['steps']==20000 and fit['selected_dev']['count']==500
    for seed in range(3):
        out=RUNS/f'sudoku-extreme-s{seed}'
        stage(f'lattice-seed{seed}-score',[PY,CODE/'trmp/lattice_score.py','--run',out,'--shard','103197'],out/'.inductor-score')
        stage(f'lattice-seed{seed}-count',[PY,CODE/'flops/trmp_lattice_count.py','--run',out],out/'.inductor-count')
    if a.warm:
        out=RUNS/'timing-seed0'
        stage('lattice-seed0-warm',[PY,CODE/'trmp/lattice_fit_warm.py','0',out,'7'],out/'.inductor')
    event(status='queue-complete')
