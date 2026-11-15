"""H4z weighted search; adapted from the frozen H1ah runner.

All state, hidden-target and saved-board checks are retained. The numerical
representation uses integer numerators over256 and a separate exact checker.
"""
from pathlib import Path
import hashlib
import json
import os
import platform
import sys
import time
FOUREYES = Path(__file__).resolve().parent
sys.path.insert(0, str(FOUREYES.parent/'ripper'))
from run import ROOT, UP, HERE, save, sha, now, captured_search_class, independent_validity
OUT = ROOT/'artifacts/research/20261114/foureyes/weak_weight_study'
PROTOCOL = FOUREYES/'weak_weight_study.json'

def run(j, gpu, p, torch, np, GS, SS, SP, dataset):
    from sparse_backend import sparse_search_class
    dest = OUT/'jobs'/j['job_id']; dest.mkdir(parents=True, exist_ok=False)
    e = j['model']; path = ROOT/e['path']; assert sha(path) == e['sha256']
    source=np.load(path);support=source['support'];weight256=source['weight256'];seed=j['seed']
    assert np.array_equal(weight256>0,support) and set(np.unique(weight256))<={0,1,4,16,256}
    J = torch.as_tensor(weight256.astype(np.float32)/256, device='cuda')
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
    from weak_weight_math import check_state
    parity = 0; hidden_target_parity = 0;integer_checks=0;integer_chains=0
    for check in range(6):
        integer_chains+=check_state(y,weight256);integer_checks+=1
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
        'weight256_bytes_sha256':hashlib.sha256(weight256.tobytes()).hexdigest(),'integer_field_checks':integer_checks,'integer_chain_states':integer_chains,
        'integer_audit_sha256':sha(FOUREYES/'weak_weight_math.py'),'edges':int(support.sum()),'used_unit':1.0,'tau':0.0,'ebar':0.0,'search_kwargs':kw,
        'preflight_state_comparisons':parity,'hidden_target_state_comparisons':hidden_target_parity,
        'move_cap':262144,'chains':4,'rows_half_open':p['rows_half_open'],'test_rows':len(sol),
        'D':ss.D,'fl_move':ss.fl_move,'torch':torch.__version__,'cuda':torch.version.cuda,
        'hardware':torch.cuda.get_device_name()}
    save(dest/'receipt.json',rec); torch.save({'W':W.cpu(),'B':B.cpu(),'weight256':torch.from_numpy(weight256)},dest/'searched_model.pt')
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
