"""Chain-count allocation at a fixed maximum chain-move envelope."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
UP = ROOT / 'upstream'
RIPPER = ROOT / 'research/20261114/ripper'
BASE = ROOT / 'artifacts/research/20261114/ripper'
OUT = ROOT / 'artifacts/research/20261114/foureyes/chain_budget'
PROTOCOL = HERE / 'chain_budget.json'
sys.path.insert(0, str(RIPPER))
from run import sha, save, now, independent_validity


def queue(gpu=None, job_id=None, status=None):
    with (OUT / 'queue.lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        jobs = json.loads((OUT / 'queue.json').read_text())
        if job_id is None:
            j = next((j for j in jobs if j['status'] == 'pending'), None)
            if j is not None:
                j.update(status='running', gpu=gpu, pid=os.getpid(), started_utc=now())
        else:
            j = next(j for j in jobs if j['job_id'] == job_id)
            j.update(status=status, updated_utc=now())
        save(OUT / 'queue.json', jobs)
        return j


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--initialize', action='store_true')
    args = parser.parse_args()
    assert platform.node() == 'foureyes'
    p = json.loads(PROTOCOL.read_text())
    if args.initialize:
        OUT.mkdir(parents=True, exist_ok=True)
        assert not (OUT / 'queue.json').exists()
        for path, digest in (p['frozen_inputs'] | p['source_sha256']).items():
            assert sha(ROOT / path) == digest
        save(OUT / 'queue.json', [dict(job_id=f's{s}-{arm}',seed=s,arm=arm,status='pending') for s in p['search_seeds'] for arm in (p['arms'] if s%2==0 else p['arms'][::-1])])
        return
    lock = (ROOT / f'artifacts/gpu{args.gpu}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu = subprocess.check_output(['nvidia-smi', f'--id={args.gpu}', '--query-gpu=uuid,name', '--format=csv,noheader'], text=True).strip()
    owners = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True)
    assert gpu.split(',')[0] not in owners, 'Unrelated GPU owner'
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), CRAM_DATA=str(UP / 'data'),
                      OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    sys.path[:0] = [str(UP), str(UP / 'experiments')]
    import torch
    import numpy as np
    import sudoku as CF
    from cram import cram_sudoku_general as GS, cram_sudoku_search as SS, cram_sudoku_sparse as SP
    from restart_backend import sparse_search_class
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    klass = sparse_search_class(GS, SP)
    for path, expected in p['source_sha256'].items():
        assert sha(ROOT / path) == expected
    sources = {str(f.relative_to(ROOT)): sha(f) for f in [Path(__file__), PROTOCOL, RIPPER / 'run.py', RIPPER / 'sparse_backend.py', HERE / 'restart_backend.py', HERE / 'restart_capture.py', UP / 'cram/cram_sudoku_general.py', UP / 'cram/cram_sudoku_search.py', UP / 'cram/cram_sudoku_sparse.py']}
    sol, free, idx = SS.load_puzzles(p['split'], p['rows_half_open'][0], p['rows_half_open'][1] - p['rows_half_open'][0])

    def same(a, b):
        assert a.keys() == b.keys()
        for key in a:
            if key == 'elapsed':
                continue
            assert torch.equal(a[key].cpu(), b[key].cpu()) if torch.is_tensor(a[key]) else a[key] == b[key], key

    def execute(job):
        seed,arm=job['seed'],job['arm']
        chains=p['chains_by_arm'][arm]
        dest = OUT / job['job_id']
        dest.mkdir(parents=True, exist_ok=False)
        for path, expected in p['frozen_inputs'].items():
            assert sha(ROOT / path) == expected
        support = np.load(BASE / 'count/k10-s0/counts.npz')['support']
        train = np.load(BASE / 'inputs/k10-s0/train.npz')['solutions'].astype(np.int64)
        pair = CF.pair_index('cuda')
        ii, jj = np.triu_indices(81, 1)
        inferred = ~(train[:, ii] == train[:, jj]).any(0)
        assert np.array_equal(inferred, support)
        W = torch.as_tensor(support, device='cuda').float()[:, None, None] * torch.eye(9, device='cuda')
        B = torch.zeros(81, 9, device='cuda')
        m = CF.prepare(W, B, torch.as_tensor(train, device='cuda'))
        assert m['tau_train'] == 0 and m['ebar'] == 0 and m['hbar'] == 1 and m['D'] == 20
        kw = dict(temp=.4, theta=.45, focused=True, C=chains, tau=0, tol=.001, ebar=0,
                  cap=4e11, check=4096, seed=seed, anneal=(.4, .31, 524288),restart=0,growth=1.)
        from restart_capture import captured_restart_class
        prekw=dict(kw,tau=-1e30)
        reference = GS.GeneralSearch(m['S'], B, sol[:16], free[:16], **prekw)
        captured = captured_restart_class(GS)(m['S'], B, sol[:16], free[:16], **prekw)
        probe = klass(m['S'], B, sol[:16], free[:16], **prekw)
        for _ in range(8):
            reference.step_check(); captured.step_check(); probe.step_check()
            same(reference.state_dict(), captured.state_dict())
            same(reference.state_dict(), probe.state_dict())
            assert torch.equal(captured.boards(), probe.boards())
        del reference, captured, probe
        ss = klass(m['S'], B, sol, free, **kw)
        inputs = p['frozen_inputs']
        rec = dict(status='running', job_id=job['job_id'],seed=seed,arm=arm,restart_period=kw['restart'],k=10, gpu=gpu, host=platform.node(), pid=os.getpid(),
                   started_utc=now(), source_sha256=sources, input_sha256=inputs,
                   bounded_native_capture_sparse_parity=True,preflight_checks=8,preflight_forced_restart_count=0, chains=chains, state_fields=len(ss.state_dict()) - 1,
                   capture_fields=2, prefix_moves=ss.t, torch=torch.__version__, cuda=torch.version.cuda,
                   cumulative_prefix_search_seconds=ss.elapsed, checkpoints=[])
        save(dest / 'receipt.json', rec)
        last = [time.monotonic()]
        total_wall = 0.0
        prefix_stopped = np.zeros(len(sol), dtype=bool)
        prefix_boards = np.zeros_like(sol)
        prefix_moves = np.zeros(len(sol), dtype=np.int64)
        for envelope in p['checkpoints'][1:]:
            assert envelope % chains == 0
            cap = envelope // chains
            assert cap % ss.check == 0
            ss.T_cap = cap
            ss.cap = cap * ss.fl_move * ss.C
            def progress(s):
                if time.monotonic() - last[0] > 2 or s.done():
                    save(dest / 'progress.json', dict(utc=now(), pid=os.getpid(), seed=seed, move=s.t, cap=s.T_cap, active=len(s.act), elapsed_s=s.elapsed))
                    last[0] = time.monotonic()
            start = time.monotonic()
            result = ss.run(progress, log_every=1 << 20)
            torch.cuda.synchronize()
            wall = time.monotonic() - start
            total_wall += wall
            # Direct integer recomputation of fields and pair-conflict energy.
            y = ss.y.long()
            J = m['S'][:, :, 0, 0]
            expected_G = torch.einsum('ij,hjv->hiv', J, torch.nn.functional.one_hot(y, 9).float())
            expected_P = (y[:, torch.as_tensor(ii[support], device='cuda')] == y[:, torch.as_tensor(jj[support], device='cuda')]).sum(1).float()
            assert torch.equal(ss.G, expected_G) and torch.equal(ss.P, expected_P)
            assert torch.equal(ss.G, ss.G.round()) and torch.equal(ss.P, ss.P.round())
            boards = ss.boards().cpu().numpy().astype(np.int8)
            stopped = ss.stopped.cpu().numpy()
            exact = (boards == sol).all(1)
            valid = independent_validity(boards, sol, free)
            moves = torch.where(ss.stopped, ss.stop_t, torch.full_like(ss.stop_t, ss.t)).cpu().numpy()
            assert np.array_equal(exact, result['solved']) and np.array_equal(exact, valid)
            assert not (stopped & ~valid).any()
            assert stopped[prefix_stopped].all()
            assert np.array_equal(boards[prefix_stopped], prefix_boards[prefix_stopped])
            assert np.array_equal(moves[prefix_stopped], prefix_moves[prefix_stopped])
            target = dest / f'moves-{cap}'
            target.mkdir()
            np.savez_compressed(target / 'boards.npz', boards=boards, solutions=sol.astype(np.int8), free=free, puzzle=idx,
                                stopped=stopped, exact=exact, valid=valid, moves=moves)
            torch.save(ss.state_dict(), target / 'state.pt')
            torch.save(dict(best_boards=ss.best_boards.cpu(), cert_boards=ss.cert_boards.cpu()), target / 'capture.pt')
            save(target / 'result.json', dict(summary=SS.summarize(result), per_puzzle=result))
            point = dict(cap=cap, maximum_chain_moves=envelope,chains=chains, restart_count_sum=int(ss.restarts.sum()), restart_count_mean=float(ss.restarts.float().mean()), terminal_move=ss.t, solved=int(exact.sum()), stopped=int(stopped.sum()), wrong_stops=int((stopped & ~valid).sum()),
                         integer_fields_and_energy_exact=True, n=len(sol), mean_moves=float(moves.mean()), segment_wall_s=wall, continuation_wall_s=total_wall,
                         cumulative_search_seconds=ss.elapsed, output_sha256={name: sha(target / name) for name in ['boards.npz', 'state.pt', 'capture.pt', 'result.json']})
            rec['checkpoints'].append(point)
            save(dest / 'receipt.json', rec)
            print(json.dumps(dict(seed=seed, **point)), flush=True)
            prefix_stopped, prefix_boards, prefix_moves = stopped.copy(), boards.copy(), moves.copy()
        rec.update(status='complete', completed_utc=now())
        save(dest / 'receipt.json', rec)

    while True:
        job = queue(gpu=args.gpu)
        if job is None:
            break
        try:
            execute(job)
        except BaseException:
            queue(job_id=job['job_id'], status='failed')
            raise
        queue(job_id=job['job_id'], status='complete')


if __name__ == '__main__':
    main()
