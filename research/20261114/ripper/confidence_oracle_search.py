"""Frozen H1aj oracle interventions on the same already-seen 5,000 puzzles."""
import argparse
import fcntl
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from confidence_oracle_common import load_protocol
from run import ROOT, UP, HERE, save, sha, now, captured_search_class, independent_validity

OUT = ROOT/'artifacts/research/20261114/ripper/confidence_oracle'
PROTOCOL = HERE/'protocol-confidence-oracle.json'


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
    manifest=json.loads((OUT/'models.json').read_text());entries=manifest['models']
    assert manifest['protocol_sha256']==sha(PROTOCOL)
    prior=ROOT/'artifacts/research/20261114/ripper/confidence_budget'
    prior_audit=json.loads((prior/'audit.json').read_text())
    old_mapping=json.loads((prior/'search_mapping.json').read_text())
    datahash=sha(UP/'data/full-test/full-all.npz')
    assert datahash==p['test_data_sha256']==old_mapping['test_data_sha256']
    known={};mapping=[];q=[]
    for seed in p['seeds']:
        oldrow=next(r for r in old_mapping['rows'] if r['seed']==seed and r['arm']=='evidence')
        directory=prior/'jobs'/oldrow['canonical_job'];receipt=json.loads((directory/'receipt.json').read_text())
        relative=str((directory/'receipt.json').relative_to(ROOT))
        assert sha(directory/'receipt.json')==prior_audit['artifact_sha256'][relative]
        assert receipt['status']=='complete' and receipt['seed']==seed and receipt['rows_half_open']==p['rows_half_open']
        assert receipt['move_cap']==262144 and receipt['backend_sha256']==p['backend_sha256'] and receipt['capture_sha256']==p['capture_sha256']
        expected=dict(temp=.4,theta=.45,focused=True,C=4,tau=0.,tol=.001,ebar=0.,cap=4e11,check=4096,seed=seed,anneal=[.4,.31,524288])
        assert receipt['search_kwargs']==expected
        base=next(e for e in entries if e['seed']==seed and e['arm']=='base');weight=np.load(ROOT/base['path'])['weight16']
        assert hashlib.sha256(weight.tobytes()).hexdigest()==receipt['weight16_bytes_sha256']
        known[seed,weight.tobytes()]={'kind':'archived','job':oldrow['canonical_job'],'directory':str(directory.relative_to(ROOT))}
    for seed in p['seeds']:
        for arm in p['arms']:
            e=next(e for e in entries if e['seed']==seed and e['arm']==arm)
            assert sha(ROOT/e['path'])==e['sha256'];z=np.load(ROOT/e['path']);weight=z['weight16'];support=z['support']
            assert np.array_equal(weight>0,support)
            key=seed,weight.tobytes();requested=f'{arm}-n1110-s{seed}'
            if key not in known:
                known[key]={'kind':'new','job':requested,'directory':str((OUT/'jobs'/requested).relative_to(ROOT))}
                q.append({'job_id':requested,'seed':seed,'n':1110,'arm':arm,'model':e,'status':'pending'})
            canonical=known[key]
            mapping.append({'requested':requested,'seed':seed,'n':1110,'arm':arm,'canonical_kind':canonical['kind'],
                'canonical_job':canonical['job'],'canonical_directory':canonical['directory'],
                'reused':canonical['kind']=='archived' or canonical['job']!=requested,
                'weight16_bytes_sha256':hashlib.sha256(weight.tobytes()).hexdigest()})
    assert not (OUT/'queue.json').exists();save(OUT/'queue.json',q)
    save(OUT/'search_mapping.json',{'utc':now(),'protocol_sha256':sha(PROTOCOL),'wrapper_sha256':sha(__file__),
        'test_data_sha256':datahash,'requested_models':len(mapping),'new_unique_searches':len(q),
        'archived_reuse_cells':sum(r['canonical_kind']=='archived' for r in mapping),
        'rule':'Identical full weights, seed, inputs, backend, capture, temperatures, focus, chains, cap and tolerance; archived baseline receipts checked explicitly.', 'rows':mapping})
    print(json.dumps({'requested':len(mapping),'new_unique_searches':len(q),'archived_reuse_cells':sum(r['canonical_kind']=='archived' for r in mapping)}))


def run(j, gpu, p, torch, np, GS, SS, SP, dataset):
    from sparse_backend import sparse_search_class
    dest = OUT/'jobs'/j['job_id']; dest.mkdir(parents=True, exist_ok=False)
    e = j['model']; path = ROOT/e['path']; assert sha(path) == e['sha256']
    source=np.load(path);support=source['support'];weight16=source['weight16'];seed=j['seed']
    assert np.array_equal(weight16>0,support) and set(np.unique(weight16))<={0,1,16}
    J = torch.as_tensor(weight16.astype(np.float32)/16, device='cuda')
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
    from confidence_integer_audit import check_state
    parity = 0; hidden_target_parity = 0;integer_checks=0;integer_chains=0
    for check in range(6):
        integer_chains+=check_state(y,weight16);integer_checks+=1
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
        elif check==4:
            for state in (x,y,z):state._moves(127);state.t+=127
    del x,y,z
    ss = klass(S,B,sol,free,**kw); ss.T_cap = 262144; ss.cap = ss.T_cap*ss.fl_move*4
    rec = {'status':'running','job_id':j['job_id'],'n':j['n'],'seed':seed,'arm':j['arm'],
        'gpu':gpu,'host':platform.node(),'pid':os.getpid(),'started_utc':now(),
        'protocol_sha256':sha(PROTOCOL),'wrapper_sha256':sha(__file__),
        'backend_sha256':sha(HERE/'sparse_backend.py'),'capture_sha256':sha(HERE/'run.py'),
        'support_file_sha256':sha(path),'support_bytes_sha256':hashlib.sha256(support.tobytes()).hexdigest(),
        'weight16_bytes_sha256':hashlib.sha256(weight16.tobytes()).hexdigest(),'integer_field_checks':integer_checks,'integer_chain_states':integer_chains,
        'integer_audit_sha256':sha(HERE/'confidence_integer_audit.py'),'edges':int(support.sum()),'used_unit':1.0,'tau':0.0,'ebar':0.0,'search_kwargs':kw,
        'preflight_state_comparisons':parity,'hidden_target_state_comparisons':hidden_target_parity,
        'move_cap':262144,'chains':4,'rows_half_open':p['rows_half_open'],'test_rows':len(sol),
        'D':ss.D,'fl_move':ss.fl_move,'torch':torch.__version__,'cuda':torch.version.cuda,
        'hardware':torch.cuda.get_device_name()}
    save(dest/'receipt.json',rec); torch.save({'W':W.cpu(),'B':B.cpu(),'weight16':torch.from_numpy(weight16)},dest/'searched_model.pt')
    last = [time.monotonic()]
    def progress(s):
        if time.monotonic()-last[0] >= .5 or s.done():
            save(dest/'progress.json',{'utc':now(),'pid':os.getpid(),'move':s.t,'cap':s.T_cap,
                 'active':len(s.act),'elapsed_s':s.elapsed}); last[0] = time.monotonic()
    start = time.monotonic(); result = ss.run(progress,log_every=65536); torch.cuda.synchronize()
    wall = time.monotonic()-start
    boards = ss.boards().cpu().numpy().astype(np.int8); exact = (boards == sol).all(1)
    valid = independent_validity(boards,sol,free)
    assert np.array_equal(exact,result['solved'])
    stopped = ss.stopped.cpu().numpy()
    moves = torch.where(ss.stopped,ss.stop_t,torch.full_like(ss.stop_t,ss.t)).cpu().numpy()
    np.savez_compressed(dest/'boards.npz',boards=boards,solutions=sol.astype(np.int8),free=free,
                        puzzle=idx,stopped=stopped,exact=exact,valid=valid,moves=moves)
    save(dest/'result.json',{'summary':SS.summarize(result),'per_puzzle':result})
    rec.update(status='complete',completed_utc=now(),wall_s=wall,terminal_move=ss.t,
        solved=int(exact.sum()),valid=int(valid.sum()),wrong_stops=int((stopped&~valid).sum()),noncanonical_valid_returns=int((valid&~exact).sum()),noncanonical_valid_certificates=int((stopped&valid&~exact).sum()),
        budget_failures=int((~stopped).sum()),mean_moves=float(moves.mean()),
        output_sha256={n:sha(dest/n) for n in ['boards.npz','result.json','searched_model.pt']})
    save(dest/'receipt.json',rec)
    print(json.dumps({'job_id':j['job_id'],'solved':rec['solved'],'wrong_stops':rec['wrong_stops'],
                     'wall_s':wall}),flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--initialize',action='store_true'); ap.add_argument('--gpu',type=int)
    a = ap.parse_args(); assert platform.node() == 'ripper'
    p = load_protocol()
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
