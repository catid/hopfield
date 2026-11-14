"""H4: rule-checked Sudoku stopping, isolated from the frozen reproduction."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
UP = ROOT / 'upstream'
OUT = ROOT / 'artifacts/research/20261114/foureyes/calibration'
CAPTURE = ROOT / 'research/20261114/ripper/run.py'
CAPTURE_SHA = 'b2ce3c3b9ff5fb9348d300b2dc02c85ddd19a6790dd3052ba34c6e53fa8900ac'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, required=True)
    ap.add_argument('--lane', type=int, default=0)
    ap.add_argument('--preflight', action='store_true')
    a = ap.parse_args()
    assert platform.node() == 'foureyes'
    lock = (ROOT / f'artifacts/gpu{a.gpu}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    query = lambda args: subprocess.check_output(args, text=True, timeout=20).strip()
    gpu = query(['nvidia-smi',f'--id={a.gpu}','--query-gpu=uuid,name,utilization.gpu,memory.used','--format=csv,noheader'])
    uuid = gpu.split(',')[0].strip()
    owners = query(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'])
    assert uuid not in owners, 'Existing compute owner'
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), CRAM_DATA=str(UP/'data'),
                      OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    sys.path[:0] = [str(UP), str(UP/'experiments'), str(HERE)]
    import numpy as np
    import torch
    import triton
    import sudoku as CF
    from cram import cram_sudoku_general as GS, cram_sudoku_search as SS
    from gate_kernel import _moves_gate
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert sha(CAPTURE) == CAPTURE_SHA
    spec = importlib.util.spec_from_file_location('frozen_h1_capture', CAPTURE)
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    Base = capture.captured_search_class(GS)
    cells = torch.arange(81, device='cuda')
    row, col = cells // 9, cells % 9
    peer = ((row[:,None] == row[None,:]) | (col[:,None] == col[None,:]) |
            ((row[:,None]//3 == row[None,:]//3) & (col[:,None]//3 == col[None,:]//3)))
    ii, jj = torch.where(torch.triu(peer, diagonal=1))
    assert len(ii) == 810
    def conflicts(y):
        return (y[:, ii] == y[:, jj]).sum(1).float()

    class Gated(Base):
        def __init__(self, *args, gate=0, **kw):
            self.gate = gate
            super().__init__(*args, **kw)

        def _moves(self, nmoves):
            H = len(self.y)
            if not H:
                return
            if self.gate:
                self.Q.copy_(conflicts(self.y))
            seed_t = torch.tensor([self.seed*1000003 + self.t//self.check], dtype=torch.int64, device=self.dev)
            temp = torch.full((H,), self.current_temp(), device=self.dev)
            _moves_gate[(H,)](self.nbr, self.R, self.Bc, self.y, self.G, self.flag, self.P, self.Q,
                self.freelist, self.fmask, self.nfree, temp, self.found, self.ycert, seed_t,
                nmoves, self.t, self.thr, self.tau+self.tol,
                MODE=2 if self.focused else 0, GATE=self.gate, N=81,V=9,D=self.D,
                NB=128,VB=16,DB=triton.next_power_of_2(self.D),num_warps=1)

    h1 = json.loads((ROOT/'research/20261114/ripper/protocol-confirmation.json').read_text())
    protocol = json.loads((HERE/'calibration.json').read_text())
    source_files = [Path(__file__), HERE/'gate_kernel.py', CAPTURE,
                    UP/'cram/cram_sudoku_general.py', UP/'cram/cram_sudoku_search.py', UP/'experiments/sudoku.py']
    sources = {str(p.relative_to(ROOT)):sha(p) for p in source_files}

    def inputs(k, seed, n=2048):
        entry = next(e for e in h1['frozen_inputs'] if e['k']==k and e['seed']==seed)
        directory = ROOT / entry['source_dir']
        for name, expected in entry['sha256'].items():
            assert sha(directory/name) == expected
        sd = torch.load(directory/'model.pt', weights_only=True, map_location='cuda')
        training = np.load(directory/'train.npz')['solutions'].astype(np.int64)
        m = CF.prepare(sd['W'],sd['B'],torch.as_tensor(training,device='cuda'))
        train_E=GS.board_energy(m['S'],sd['B'],torch.as_tensor(training,device='cuda'))
        m['tau_min']=float(train_E.min());m['tau_p05']=float(torch.quantile(train_E,.05))
        fitted = json.loads((directory/'fit.json').read_text())
        assert m['tau_train']==fitted['tau'] and m['hbar']==fitted['hbar']
        sol, free, idx = SS.load_puzzles('dev10k',2560,n)
        kw = dict(temp=.4*m['hbar'],theta=.45*m['hbar'],focused=True,C=4,
                  tau=m['tau_train'],tol=.001,ebar=m['ebar'],check=4096,seed=seed,
                  anneal=(.4*m['hbar'],.31*m['hbar'],524288),restart=0)
        return entry, sd, m, sol, free, idx, kw

    if a.preflight:
        entry, sd, m, sol, free, idx, kw = inputs(17,0,16)
        original = GS.GeneralSearch(m['S'],sd['B'],sol,free,**kw)
        zero = Gated(m['S'],sd['B'],sol,free,gate=0,**kw)
        for _ in range(4):
            original.step_check();zero.step_check()
            for key in original.STATE:
                assert torch.equal(getattr(original,key),getattr(zero,key)), ('original_gate0',key)
        never = dict(kw,tau=-1e30)
        original = GS.GeneralSearch(m['S'],sd['B'],sol,free,**never)
        guarded = Gated(m['S'],sd['B'],sol,free,gate=1,**never)
        for nmoves in (1,7,64,4096):
            original._moves(nmoves);guarded._moves(nmoves)
            for key in original.STATE:
                if key!='Q':
                    assert torch.equal(getattr(original,key),getattr(guarded,key)), ('guard_no_stop',nmoves,key)
            assert torch.equal(guarded.Q,conflicts(guarded.y)), ('counter',nmoves)
        one_free = np.zeros_like(free);one_free[:,0]=True
        for mode in (1,2):
            simple = Gated(m['S'],sd['B'],sol,one_free,gate=mode,
                           **dict(kw,tau=1e30,temp=1e6,anneal=None,focused=False))
            simple._moves(256)
            assert bool((simple.found>=0).all())
            assert bool((conflicts(simple.ycert)==0).all())
            assert bool((simple.ycert.long()==simple.tgt).all())
            assert torch.equal(simple.Q,conflicts(simple.y))
        timing = {}
        for mode in (0,1):
            probe = Gated(m['S'],sd['B'],sol,free,gate=mode,**never)
            probe._moves(4096);torch.cuda.synchronize()
            ts=time.perf_counter()
            for _ in range(8):probe._moves(4096)
            torch.cuda.synchronize();timing[str(mode)]=(time.perf_counter()-ts)/8
        save(OUT/'preflight.json',{'status':'validated','utc':now(),'source_sha256':sources,
             'protocol_sha256':sha(HERE/'calibration.json'),'base_state_tensors':len(original.STATE),
             'original_gate0_four_checks_exact':True,'no_stop_gate1_exact_except_counter':True,
             'counter_independently_recomputed':True,'one_free_cell_modes1and2_all_valid':True,
             'seconds_per4096moves_64chains':timing,'gate_overhead_ratio':timing['1']/timing['0']})
        print(json.dumps({'status':'preflight_passed','timing':timing}),flush=True)
        return

    proof=json.loads((OUT/'preflight.json').read_text())
    assert proof['source_sha256']==sources and proof['protocol_sha256']==sha(HERE/'calibration.json')
    jobs=[(k,s,arm) for k in (17,21) for s in (0,1,2,3,4,5) for arm in protocol['arms']]
    assigned=jobs[a.lane::4]
    lane={'status':'running','pid':os.getpid(),'gpu':a.gpu,'gpu_info':gpu,'started_utc':now(),'jobs':assigned}
    save(OUT/f'lane{a.lane}.json',lane)
    for k, seed, arm in assigned:
        dest=OUT/'jobs'/f'k{k}-s{seed}-{arm}'
        assert not dest.exists(),f'Refusing overwrite: {dest}'
        entry,sd,m,sol,free,idx,kw=inputs(k,seed)
        if arm=='energy_min':kw['tau']=m['tau_min']
        if arm=='energy_p05':kw['tau']=m['tau_p05']
        search=Gated(m['S'],sd['B'],sol,free,gate=1 if arm=='energy_AND_valid' else 0,**kw)
        search.T_cap=protocol['move_cap_per_chain']
        search.cap=search.T_cap*search.fl_move*search.C
        rec={'status':'running','job_id':dest.name,'host':platform.node(),'pid':os.getpid(),
             'k':k,'seed':seed,'arm':arm,'started_utc':now(),'gpu':gpu,
             'protocol_sha256':sha(HERE/'calibration.json'),'source_sha256':sources,
             'input_sha256':entry['sha256'],'input_directory':entry['source_dir'],
             'development_sha256':sha(UP/'data/frm/dev10k/dev.npz'),
             'training_statistics':{z:m[z] for z in ('g','tau_train','tau_min','tau_p05','ebar','hbar','D')},
             'torch':torch.__version__,'cuda':torch.version.cuda,'triton':triton.__version__,
             'move_cap':search.T_cap,'chains':4,'legacy_fl_move_excludes_gate':search.fl_move}
        save(dest/'receipt.json',rec)
        last=[time.monotonic()]
        def progress(s):
            if time.monotonic()-last[0]>2 or s.done():
                save(dest/'progress.json',{'utc':now(),'pid':os.getpid(),'move':s.t,'cap':s.T_cap,'active':len(s.act)})
                last[0]=time.monotonic()
        begin=time.perf_counter();result=search.run(progress);torch.cuda.synchronize();wall=time.perf_counter()-begin
        boards=search.boards().cpu().numpy().astype(np.int8)
        valid=capture.independent_validity(boards,sol,free)
        exact=(boards==sol).all(1)
        stopped=search.stopped.cpu().numpy()
        assert np.array_equal(valid,exact) and np.array_equal(exact,np.asarray(result['solved']))
        if arm=='energy_AND_valid':assert not (stopped & ~valid).any()
        moves=torch.where(search.stopped,search.stop_t,torch.full_like(search.stop_t,search.t)).cpu().numpy()
        np.savez_compressed(dest/'boards.npz',boards=boards,solutions=sol.astype(np.int8),free=free,puzzle=idx,
                            valid=valid,exact=exact,stopped=stopped,moves=moves)
        save(dest/'result.json',{'summary':SS.summarize(result),'per_puzzle':result,
                               'flop_note':'Legacy move proxy excludes symbolic gate; not total arithmetic'})
        rec.update(status='complete',completed_utc=now(),wall_s=wall,terminal_move=search.t,
                   solved=int(exact.sum()),valid=int(valid.sum()),wrong_stops=int((stopped&~valid).sum()),
                   stopped=int(stopped.sum()),budget_failures=int((~stopped).sum()),mean_moves=float(moves.mean()),
                   output_sha256={n:sha(dest/n) for n in ('boards.npz','result.json')})
        save(dest/'receipt.json',rec)
        print(json.dumps({'event':'complete','job':dest.name,'solved':rec['solved'],'wall_s':wall}),flush=True)
    lane.update(status='complete',completed_utc=now());save(OUT/f'lane{a.lane}.json',lane)


if __name__=='__main__':
    main()
