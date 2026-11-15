"""H4aa: remove weak constraints instead of only weakening positive penalties."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
from weak_weight_study import ROOT, HERE, RIPPER, sha, save

OUT = ROOT/'artifacts/research/20261114/foureyes/strong_support_study'
PUBLIC = ROOT/'results/research/foureyes/strong_support_study'
PROTOCOL = HERE/'strong_support_study.json'
PARENT = ROOT/'results/research/foureyes/weak_weight_study'
MODELS = ROOT/'results/research/foureyes/canonical_confirm/models.npz'


def verify(p):
    for path, digest in (p['source_sha256'] | p['input_sha256']).items():
        assert sha(ROOT/path) == digest, path


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists()
    source = [HERE/n for n in ['strong_support_study.py', 'audit_strong_support_study.py',
              'weak_weight_study.py', 'weak_weight_search.py', 'weak_weight_math.py']]
    source += [RIPPER/n for n in ['run.py', 'sparse_backend.py', 'confidence_weight_search.py', 'bootstrap_two_way.py']]
    source += [ROOT/'upstream/cram'/n for n in ['cram_sudoku_general.py', 'cram_sudoku_search.py', 'cram_sudoku_sparse.py']]
    inputs = [MODELS, PARENT/'summary.json', PARENT/'audited-outcome-bits.npz',
              HERE/'weak_weight_study.json', ROOT/'upstream/data/full-test/full-all.npz']
    p = dict(id='H4aa-positive-versus-removed-weak-constraints', frozen_utc=datetime.now(timezone.utc).isoformat(),
             scope='Post-hoc mechanism test on all64 H4y fits and same5000 already-seen H4y/H4z puzzles. No fresh-generalization claim.',
             arms=['weak256', 'strongonly'], banks=64, rows_half_open=[410000,415000], search_seed=0,
             rule='Same median-positive-J/4 split as H4z. Archived weak256 has weak1/256,strong1; strongonly deletes every weak edge and keeps strong1. No labels or geometry in assignment.',
             primary='strongonly accuracy minus weak256 >=5pp ANDzero invalidstrongonly certificates.',
             decision='5000paired two-way bootstrap draws,seed20261116041. Supported if95lower>=5 andzero wrongstops; falsified ifupper<5 orany wrongstop; otherwiseinconclusive.',
             diagnostics='Exact/valid/certified/capped outcomes, missingtrue/extra edges, canonical zero-energy compatibility and given-given infeasibility. No graph-based selection or filtering.',
             inference='Unchanged H4z settings:4chains,seed0,262144moves,temperature.4->.31,theta.45,tau0,tol.001,ebar0. Deletion changes zero set; empirical no-wrongstop doesnotprove soundness.',
             numerical_gate='Inherited H4z four-prefix gate plus all per-new-search general/sparse/hidden-target and integer checks. Strongonly numerators{0,256} are exactly representable; no weakpositiveenergy remains.',
             budget=dict(search_phase_wall_seconds=180,total_gpu_minutes_upper_bound=12),
             source_sha256={str(x.relative_to(ROOT)):sha(x) for x in source},
             input_sha256={str(x.relative_to(ROOT)):sha(x) for x in inputs})
    save(PROTOCOL,p); print(json.dumps(dict(protocol_sha256=sha(PROTOCOL))))


def prepare():
    import numpy as np
    p=json.loads(PROTOCOL.read_text());verify(p);assert not OUT.exists();OUT.mkdir(parents=True)
    parent=json.loads((PARENT/'summary.json').read_text())
    assert parent['status']=='validated' and parent['gate_launch']['status']=='complete'
    with np.load(MODELS,allow_pickle=False) as z:coeff=z['canonical_weak_coefficients']
    rows=[];queue=[];known={}
    for bank,j in enumerate(coeff):
        cut=np.median(j[j>0])/4
        for arm,weak in [('weak256',1),('strongonly',0)]:
            w=np.where(j>0,np.where(j>=cut,256,weak),0).astype(np.uint16)
            path=OUT/'models'/f'b{bank:02d}-{arm}.npz';path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,support=w>0,weight256=w)
            if arm=='weak256':
                old=next(r for r in parent['rows']if r['bank']==bank and r['arm']==arm)
                assert old['kind']=='new'
                key=('archived',old['job'])
            else:
                ident=w.tobytes()
                if ident not in known:
                    job=f'b{bank:02d}-strongonly';known[ident]=job
                    queue.append(dict(job_id=job,seed=0,n=600,arm=arm,status='pending',model=dict(path=str(path.relative_to(ROOT)),sha256=sha(path))))
                key=('new',known[ident])
            rows.append(dict(bank=bank,arm=arm,kind=key[0],job=key[1],model_path=str(path.relative_to(ROOT)),model_sha256=sha(path),weight256_sha256=hashlib.sha256(w.tobytes()).hexdigest()))
    save(OUT/'mapping.json',dict(protocol_sha256=sha(PROTOCOL),rows=rows));save(OUT/'queue.json',queue)
    print(json.dumps(dict(logical_cells=len(rows),unique_new_searches=len(queue))))


def search(gpu):
    from weak_weight_study import setup
    lock,torch,np,GS,SS,SP=setup(gpu)
    p=json.loads(PROTOCOL.read_text());verify(p)
    a=json.loads((OUT/'source-audit.json').read_text());assert a['status']=='validated' and a['protocol_sha256']==sha(PROTOCOL)
    import confidence_weight_search as queue
    import weak_weight_search as runner
    queue.OUT=OUT;runner.OUT=OUT;runner.PROTOCOL=PROTOCOL
    data=SS.load_puzzles('full-all',410000,5000)
    while (job:=queue.claim(gpu)) is not None:
        try:runner.run(job,gpu,p,torch,np,GS,SS,SP,data)
        except BaseException:queue.claim(gpu,job['job_id'],'failed');raise
        queue.claim(gpu,job['job_id'],'complete')


def launch():
    import weak_weight_study as base
    # The shared launcher starts Path(__file__) of its own module; point it at
    # this wrapper while retaining the frozen launcher implementation.
    base.OUT=OUT;base.PROTOCOL=PROTOCOL;base.verify=verify;base.__file__=__file__
    base.launch('search')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','search','launch-search']);ap.add_argument('--gpu',type=int);a=ap.parse_args()
    if a.action=='search':search(a.gpu)
    elif a.action=='launch-search':launch()
    else:globals()[a.action]()
