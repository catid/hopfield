"""Frozen H1q search: first uniform-mask stream versus original N300 supports."""
import argparse
import fcntl
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from run import ROOT, UP, HERE, save, sha, now, captured_search_class, independent_validity

OUT = ROOT/'artifacts/research/20261114/ripper/masks'
PROTOCOL = HERE/'protocol-masks.json'


def claim(gpu, job_id=None, status=None):
    with (OUT/'queue.lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX); q = json.loads((OUT/'queue.json').read_text())
        if job_id is None:
            j = next((j for j in q if j['status']=='pending'), None)
            if j is not None: j.update(status='running', gpu=gpu, pid=os.getpid(), started_utc=now())
        else:
            j = next(j for j in q if j['job_id']==job_id); j.update(status=status, completed_utc=now())
        save(OUT/'queue.json', q); return j


def initialize(p, np):
    entries = []
    oldpath = ROOT/p['official_models']
    assert sha(oldpath) == p['official_models_sha256']
    old = json.loads(oldpath.read_text())
    for seed in p['seeds']:
        official = next(e for e in old['rows'] if e['seed']==seed and e['n']==300)
        entries.append(dict(official,arm='official'))
        firstpath = OUT/f's{seed}/first_stream.json'
        first = json.loads(firstpath.read_text())
        assert first['protocol_sha256'] == sha(PROTOCOL) and first['draw'] == 0
        assert first['wrapper_sha256'] == sha(HERE/'mask_count.py')
        entries.append({'seed':seed,'n':300,'arm':'uniform','path':first['model_path'],
            'sha256':first['model_sha256'],'support_bytes_sha256':first['support_bytes_sha256']})
    q = []; mapping = []; known = {}
    datahash = sha(UP/'data/full-test/full-all.npz')
    for seed in p['seeds']:
        for n in p['search_sizes']:
            for arm in p['arms']:
                e = next(e for e in entries if (e['seed'], e['n'], e['arm']) == (seed,n,arm))
                assert sha(ROOT/e['path']) == e['sha256']
                support = np.load(ROOT/e['path'])['support']
                key = (seed, support.tobytes())
                requested = f'{arm}-n{n}-s{seed}'
                if key not in known:
                    known[key] = requested
                    q.append({'job_id':requested, 'seed':seed, 'n':n, 'arm':arm, 'model':e, 'status':'pending'})
                mapping.append({'requested':requested, 'seed':seed, 'n':n, 'arm':arm,
                    'canonical_job':known[key], 'reused':known[key]!=requested,
                    'support_bytes_sha256':hashlib.sha256(support.tobytes()).hexdigest()})
    assert len(entries) == 12
    assert not (OUT/'queue.json').exists()
    save(OUT/'queue.json',q)
    save(OUT/'search_mapping.json', {'utc':now(), 'protocol_sha256':sha(PROTOCOL),
        'wrapper_sha256':sha(__file__), 'test_data_sha256':datahash, 'requested_models':len(mapping),
        'unique_searches':len(q), 'rule':'Same support bytes and seed; B0, unit1, tau0, ebar0 and all other settings fixed by protocol.', 'rows':mapping})
    print(json.dumps({'requested':len(mapping),'unique':len(q)}))


def run(j, gpu, p, torch, np, GS, SS, SP, dataset):
    from sparse_backend import sparse_search_class
    dest = OUT/'jobs'/j['job_id']; dest.mkdir(parents=True, exist_ok=False)
    e = j['model']; path = ROOT/e['path']; assert sha(path) == e['sha256']
    support = np.load(path)['support']; seed = j['seed']
    J = torch.as_tensor(support.astype(np.float32), device='cuda')
    W = J[:,None,None]*torch.eye(9,device='cuda'); B = torch.zeros(81,9,device='cuda')
    iu = torch.triu_indices(81,81,1,device='cuda')
    S = torch.zeros(81,81,9,9,device='cuda')
    S[iu[0],iu[1]] = W; S[iu[1],iu[0]] = W.transpose(1,2)
    # No calibration routine is called. Literal constants apply to both arms.
    kw = dict(temp=.4, theta=.45, focused=True, C=4, tau=0.0, tol=.001, ebar=0.0,
              cap=4e11, check=4096, seed=seed, anneal=(.4,.31,524288))
    sol,free,idx = dataset; klass = sparse_search_class(GS,SP)
    x = captured_search_class(GS)(S,B,sol[:16],free[:16],**kw)
    y = klass(S,B,sol[:16],free[:16],**kw)
    altered = sol[:16].copy(); rng = np.random.default_rng(20261118+seed)
    altered[free[:16]] = rng.integers(0,9,free[:16].sum())
    z = klass(S,B,altered,free[:16],**kw)
    parity = 0; hidden_target_parity = 0
    for check in range(5):
        xs,ys,zs = x.state_dict(),y.state_dict(),z.state_dict()
        assert xs.keys() == ys.keys() == zs.keys()
        for key,u in xs.items():
            assert torch.equal(u,ys[key]) if torch.is_tensor(u) else u == ys[key], key
            parity += 1
            if key not in ('tgt','ok','bestok'):
                assert torch.equal(ys[key],zs[key]) if torch.is_tensor(ys[key]) else ys[key] == zs[key], key
                hidden_target_parity += 1
        assert torch.equal(x.best_boards,y.best_boards) and torch.equal(x.cert_boards,y.cert_boards)
        assert torch.equal(y.best_boards,z.best_boards) and torch.equal(y.cert_boards,z.cert_boards)
        if check < 4: x.step_check(); y.step_check(); z.step_check()
    del x,y,z
    ss = klass(S,B,sol,free,**kw); ss.T_cap = 262144; ss.cap = ss.T_cap*ss.fl_move*4
    rec = {'status':'running','job_id':j['job_id'],'n':j['n'],'seed':seed,'arm':j['arm'],
        'gpu':gpu,'host':platform.node(),'pid':os.getpid(),'started_utc':now(),
        'protocol_sha256':sha(PROTOCOL),'wrapper_sha256':sha(__file__),
        'backend_sha256':sha(HERE/'sparse_backend.py'),'capture_sha256':sha(HERE/'run.py'),
        'support_file_sha256':sha(path),'support_bytes_sha256':hashlib.sha256(support.tobytes()).hexdigest(),
        'edges':int(support.sum()),'used_unit':1.0,'tau':0.0,'ebar':0.0,'search_kwargs':kw,
        'preflight_state_comparisons':parity,'hidden_target_state_comparisons':hidden_target_parity,
        'move_cap':262144,'chains':4,'rows_half_open':p['rows_half_open'],'test_rows':len(sol),
        'D':ss.D,'fl_move':ss.fl_move,'torch':torch.__version__,'cuda':torch.version.cuda,
        'hardware':torch.cuda.get_device_name()}
    save(dest/'receipt.json',rec); torch.save({'W':W.cpu(),'B':B.cpu()},dest/'searched_model.pt')
    last = [time.monotonic()]
    def progress(s):
        if time.monotonic()-last[0] >= .5 or s.done():
            save(dest/'progress.json',{'utc':now(),'pid':os.getpid(),'move':s.t,'cap':s.T_cap,
                 'active':len(s.act),'elapsed_s':s.elapsed}); last[0] = time.monotonic()
    start = time.monotonic(); result = ss.run(progress,log_every=65536); torch.cuda.synchronize()
    wall = time.monotonic()-start
    boards = ss.boards().cpu().numpy().astype(np.int8); exact = (boards == sol).all(1)
    valid = independent_validity(boards,sol,free)
    assert np.array_equal(exact,result['solved']) and np.array_equal(exact,valid)
    stopped = ss.stopped.cpu().numpy()
    moves = torch.where(ss.stopped,ss.stop_t,torch.full_like(ss.stop_t,ss.t)).cpu().numpy()
    np.savez_compressed(dest/'boards.npz',boards=boards,solutions=sol.astype(np.int8),free=free,
                        puzzle=idx,stopped=stopped,exact=exact,valid=valid,moves=moves)
    save(dest/'result.json',{'summary':SS.summarize(result),'per_puzzle':result})
    rec.update(status='complete',completed_utc=now(),wall_s=wall,terminal_move=ss.t,
        solved=int(exact.sum()),valid=int(valid.sum()),wrong_stops=int((stopped&~exact).sum()),
        budget_failures=int((~stopped).sum()),mean_moves=float(moves.mean()),
        output_sha256={n:sha(dest/n) for n in ['boards.npz','result.json','searched_model.pt']})
    save(dest/'receipt.json',rec)
    print(json.dumps({'job_id':j['job_id'],'solved':rec['solved'],'wrong_stops':rec['wrong_stops'],
                     'wall_s':wall}),flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--initialize',action='store_true'); ap.add_argument('--gpu',type=int)
    a = ap.parse_args(); assert platform.node() == 'ripper'
    p = json.loads(PROTOCOL.read_text())
    if a.initialize:
        import numpy as np
        initialize(p,np); return
    lock = (ROOT/f'artifacts/gpu{a.gpu}.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    uuid = subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    owners = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    assert uuid not in owners
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), CRAM_DATA=str(UP/'data'),
                      OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    sys.path.insert(0,str(UP))
    import torch
    import numpy as np
    from cram import cram_sudoku_general as GS, cram_sudoku_search as SS, cram_sudoku_sparse as SP
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32=False
    assert sha(HERE/'sparse_backend.py') == p['backend_sha256'] and sha(HERE/'run.py') == p['capture_sha256']
    mapping = json.loads((OUT/'search_mapping.json').read_text())
    assert mapping['test_data_sha256'] == sha(UP/'data/full-test/full-all.npz')
    lo,hi = p['rows_half_open']; dataset = SS.load_puzzles('full-all',lo,hi-lo)
    while True:
        j = claim(a.gpu)
        if j is None: break
        try: run(j,a.gpu,p,torch,np,GS,SS,SP,dataset)
        except BaseException: claim(a.gpu,j['job_id'],'failed'); raise
        claim(a.gpu,j['job_id'],'complete')


if __name__ == '__main__': main()
