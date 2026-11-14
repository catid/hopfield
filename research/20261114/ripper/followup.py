"""Frozen H1 screen; read-only pinned source and archived inputs, isolated outputs."""
import argparse
import datetime
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE_OUT = ROOT / 'artifacts/research/20261114/ripper'
OUT = BASE_OUT
PROTOCOL_PATH = HERE / 'protocol.json'
UP = ROOT / 'upstream'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def projection(W, B):
    import torch
    v = W.shape[-1]
    mask = torch.eye(v, dtype=torch.bool, device=W.device)
    d = W[:, mask].mean(-1)
    o = W[:, ~mask].mean(-1)
    wp = torch.where(mask[None], d[:, None, None], o[:, None, None])
    return wp.contiguous(), B.mean(-1, keepdim=True).expand_as(B).contiguous()


def math_checks():
    import torch
    g = torch.Generator().manual_seed(9182)
    w = torch.randn(4, 3, 3, generator=g, dtype=torch.float64)
    b = torch.randn(5, 3, generator=g, dtype=torch.float64)
    wp, bp = projection(w, b)
    perms = [torch.tensor(p) for p in itertools.permutations(range(3))]
    literal = torch.stack([w[:, p][:, :, p] for p in perms]).mean(0)
    literal_b = torch.stack([b[:, p] for p in perms]).mean(0)
    assert torch.allclose(wp, literal, atol=1e-14, rtol=0)
    assert torch.allclose(bp, literal_b, atol=1e-14, rtol=0)
    w9 = torch.randn(7, 9, 9, generator=g, dtype=torch.float64)
    b9 = torch.randn(5, 9, generator=g, dtype=torch.float64)
    p9, q9 = projection(w9, b9)
    for _ in range(20):
        p = torch.randperm(9, generator=g)
        assert torch.equal(p9, p9[:, p][:, :, p])
        assert torch.equal(q9, q9[:, p])
    pp, qq = projection(p9, q9)
    assert torch.allclose(pp, p9, atol=1e-14, rtol=0)
    assert torch.allclose(qq, q9, atol=1e-14, rtol=0)
    return {'literal_3digit_group_average': True, 'invariance_20_permutations_9digits': True,
            'idempotent_float64_tolerance': 1e-14}


def captured_search_class(GS):
    import torch
    class Captured(GS.GeneralSearch):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.best_boards = torch.zeros(self.NQ, 81, dtype=torch.int32, device=self.dev)
            self.cert_boards = torch.zeros_like(self.best_boards)

        def step_check(self):
            # Same pinned Search.step_check, with two board-only copies added.
            self._moves(self.check)
            self.t += self.check
            if self.t % 4096 == 0:
                self._periodic_refresh()
            nA, C = len(self.act), self.C
            right = (self.y.long() == self.tgt).all(1).view(nA, C)
            mn, am = self.P.view(nA, C).min(1)
            better = mn < self.bestP[self.act]
            self.best_boards[self.act[better]] = self.y.view(nA, C, 81)[better, am[better]]
            self.bestP[self.act] = torch.where(better, mn, self.bestP[self.act])
            self.bestok[self.act] = torch.where(better, right.gather(1, am[:, None]).squeeze(1), self.bestok[self.act])
            fu = self.found.view(nA, C).long()
            fu = torch.where(fu >= 0, fu, torch.full_like(fu, 1 << 60))
            first, fc = fu.min(1)
            anyc = first < (1 << 60)
            if bool(anyc.any()):
                rc = (self.ycert.long() == self.tgt).all(1).view(nA, C)
                a = self.act[anyc]
                self.cert_boards[a] = self.ycert.view(nA, C, 81)[anyc, fc[anyc]]
                self.ok[a] = rc[anyc].gather(1, fc[anyc][:, None]).squeeze(1)
                self.stop_t[a] = first[anyc]
                self.stopped[a] = True
            assert self.restart == 0
            if bool(self.stopped[self.act].float().mean() > 0.25):
                self._compact()

        def boards(self):
            return torch.where(self.stopped[:, None], self.cert_boards, self.best_boards)
    return Captured


def independent_validity(boards, sol, free):
    import numpy as np
    b = boards.reshape(-1, 9, 9)
    want = np.arange(9)
    rows = (np.sort(b, axis=2) == want).all((1, 2))
    cols = (np.sort(b, axis=1) == want[None, :, None]).all((1, 2))
    boxes = b.reshape(-1, 3, 3, 3, 3).transpose(0, 1, 3, 2, 4).reshape(-1, 9, 9)
    boxes = (np.sort(boxes, axis=2) == want).all((1, 2))
    clues = ((boards == sol) | free).all(1)
    return rows & cols & boxes & clues


def execute(k, seed, arm, gpu, protocol, GS, SS, CF, torch, np):
    job_id = f'k{k}-s{seed}-{arm}'
    dest = OUT / 'jobs' / job_id
    if (dest / 'receipt.json').exists():
        old = json.loads((dest / 'receipt.json').read_text())
        if old['status'] == 'complete':
            raise RuntimeError(f'Refusing duplicate completed job: {job_id}')
    dest.mkdir(parents=True, exist_ok=True)
    entry = next(x for x in protocol['frozen_inputs'] if x['k'] == k and x['seed'] == seed)
    src = ROOT / entry['input_dir']
    for name, expected in entry['sha256'].items():
        assert sha(src / name) == expected
    sd = torch.load(src / 'model.pt', weights_only=True, map_location='cuda')
    W, B = sd['W'], sd['B']
    tr = np.load(src / 'train.npz')
    training = torch.as_tensor(tr['solutions'].astype(np.int64), device='cuda')
    original_hbar = float(W[W > 0].mean())
    if arm != 'original':
        W, B = projection(W, B)
    m = CF.prepare(W, B, training)
    if arm == 'original':
        fitted = json.loads((src / 'fit.json').read_text())
        assert m['tau_train'] == fitted['tau'] and m['hbar'] == fitted['hbar']
    unit = original_hbar if arm == 'projected_original_hbar' else m['hbar']
    lo, hi = protocol['rows_half_open']
    sol, free, idx = SS.load_puzzles('dev10k', lo, hi-lo)
    klass = captured_search_class(GS)
    kw = dict(temp=.40*unit, theta=.45*unit, focused=True, C=4,
              tau=m['tau_train'], tol=.001, ebar=m['ebar'], cap=4e11,
              check=4096, seed=seed, anneal=(.40*unit,.31*unit,524288))
    ss = klass(m['S'], B, sol, free, **kw)
    ss.T_cap = protocol['move_cap_per_chain']
    ss.cap = ss.T_cap * ss.fl_move * ss.C
    scalar_stats = {z: m[z] for z in ('g','ebar','tau_train','hbar','D')}
    rec = {'status':'running','job_id':job_id,'k':k,'seed':seed,'arm':arm,'gpu':gpu,
           'host':platform.node(),'pid':os.getpid(),'started_utc':now(),
           'protocol_sha256':sha(PROTOCOL_PATH),'wrapper_sha256':sha(__file__),
           'input_sha256':entry['sha256'],'development_sha256':sha(UP/'data/frm/dev10k/dev.npz'),
           'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [UP/'experiments/sudoku.py',UP/'cram/cram_sudoku_general.py',UP/'cram/cram_sudoku_search.py']},
           'training_statistics':scalar_stats,'original_hbar':original_hbar,'used_hbar':unit,
           'move_cap':ss.T_cap,'chains':4,'check':4096,'fl_move':ss.fl_move,
           'torch':torch.__version__,'cuda':torch.version.cuda,'hardware':torch.cuda.get_device_name(),
           'training_rows':len(training),'test_rows':len(sol),'development_rows_half_open':[lo,hi]}
    save(dest/'receipt.json',rec)
    torch.save({'W':W.cpu(),'B':B.cpu()},dest/'searched_model.pt')
    last=[time.monotonic()]
    def progress(s):
        if time.monotonic()-last[0] >= 5 or s.done():
            save(dest/'progress.json',{'utc':now(),'pid':os.getpid(),'move':s.t,'cap':s.T_cap,'active':len(s.act),'elapsed_s':s.elapsed})
            last[0]=time.monotonic()
    start=time.monotonic()
    result=ss.run(progress,log_every=65536)
    torch.cuda.synchronize()
    wall=time.monotonic()-start
    boards=ss.boards().cpu().numpy().astype(np.int8)
    exact=(boards==sol).all(1)
    valid=independent_validity(boards,sol,free)
    assert np.array_equal(exact,np.asarray(result['solved']))
    assert np.array_equal(valid,exact), 'Valid alternate solution or checker mismatch'
    stopped=ss.stopped.cpu().numpy()
    moves=torch.where(ss.stopped,ss.stop_t,torch.full_like(ss.stop_t,ss.t)).cpu().numpy()
    np.savez_compressed(dest/'boards.npz',boards=boards,solutions=sol.astype(np.int8),free=free,puzzle=idx,
                        stopped=stopped,exact=exact,valid=valid,moves=moves)
    save(dest/'result.json',{'summary':SS.summarize(result),'per_puzzle':result})
    rec.update(status='complete',completed_utc=now(),wall_s=wall,
               terminal_move=ss.t,solved=int(exact.sum()),valid=int(valid.sum()),
               stopped=int(stopped.sum()),wrong_stops=int((stopped&~exact).sum()),
               budget_failures=int((~stopped).sum()),mean_moves=float(moves.mean()),
               output_sha256={name:sha(dest/name) for name in ['boards.npz','result.json','searched_model.pt']})
    save(dest/'receipt.json',rec)
    print(json.dumps({'event':'job_complete',**{x:rec[x] for x in ['job_id','solved','wrong_stops','budget_failures','wall_s','mean_moves']}}),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',choices=['confirmation','lowdata'],required=True);ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--lane',type=int,required=True);ap.add_argument('--preflight',action='store_true');a=ap.parse_args()
    global OUT, PROTOCOL_PATH
    OUT = BASE_OUT / a.study
    PROTOCOL_PATH = HERE / f'protocol-{a.study}.json'
    assert platform.node()=='ripper', 'This worker is assigned to ripper only'
    lock=(ROOT/f'artifacts/gpu{a.gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    q=lambda args:subprocess.check_output(args,text=True,timeout=20).strip()
    gpuinfo=q(['nvidia-smi',f'--id={a.gpu}','--query-gpu=uuid,name,utilization.gpu,memory.used','--format=csv,noheader'])
    uuid=gpuinfo.split(',')[0].strip();owners=q(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'])
    assert uuid not in owners, 'GPU has another compute owner'
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),CRAM_DATA=str(UP/'data'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    sys.path[:0]=[str(UP),str(UP/'experiments')]
    import torch
    import numpy as np
    import sudoku as CF
    from cram import cram_sudoku_general as GS,cram_sudoku_search as SS
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    protocol=json.loads(PROTOCOL_PATH.read_text())
    checks=math_checks()
    if a.preflight:
        src=BASE_OUT/'inputs/k17-s0';sd=torch.load(src/'model.pt',weights_only=True,map_location='cuda');tr=np.load(src/'train.npz')
        m=CF.prepare(sd['W'],sd['B'],torch.as_tensor(tr['solutions'].astype(np.int64),device='cuda'))
        sol,free,_=SS.load_puzzles('dev10k',0,16)
        kw=dict(temp=.4*m['hbar'],theta=.45*m['hbar'],C=4,tau=m['tau_train'],ebar=m['ebar'],check=4096,seed=97)
        x=GS.GeneralSearch(m['S'],sd['B'],sol,free,**kw);y=captured_search_class(GS)(m['S'],sd['B'],sol,free,**kw)
        for _ in range(4):
            x.step_check();y.step_check()
            for key in x.STATE:
                assert torch.equal(getattr(x,key),getattr(y,key)),key
        checks['capture_20_STATE_tensors_equal_four_checks']=True
        checks['source_sha256']=sha(__file__);checks['protocol_sha256']=sha(PROTOCOL_PATH);checks['utc']=now()
        save(BASE_OUT/'followup-preflight.json',checks);print(json.dumps(checks));return
    proof=json.loads((BASE_OUT/'followup-preflight.json').read_text());assert proof['source_sha256']==sha(__file__)
    jobs=[(k,s,arm) for k in protocol['primary_k'] for s in protocol['training_seeds'] for arm in protocol['arms']]
    assigned=[j for i,j in enumerate(jobs) if i%4==a.lane]
    save(OUT/f'lane{a.lane}.json',{'status':'running','pid':os.getpid(),'gpu':a.gpu,'gpu_info':gpuinfo,'jobs':assigned,'started_utc':now()})
    for k,s,arm in assigned:
        execute(k,s,arm,a.gpu,protocol,GS,SS,CF,torch,np)
    save(OUT/f'lane{a.lane}.json',{'status':'complete','pid':os.getpid(),'gpu':a.gpu,'jobs':assigned,'completed_utc':now()})


if __name__=='__main__':
    main()
