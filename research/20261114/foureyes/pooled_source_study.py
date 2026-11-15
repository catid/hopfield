"""H4ac: majority votes versus pooling the same noisy training observations."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from weak_weight_study import ROOT,HERE,RIPPER,sha,save,setup

OUT=ROOT/'artifacts/research/20261114/foureyes/pooled_source_study'
PUBLIC=ROOT/'results/research/foureyes/pooled_source_study'
PROTOCOL=HERE/'pooled_source_study.json'
PARENT=ROOT/'artifacts/research/20261114/foureyes/canonical_confirm'
MODELS=ROOT/'results/research/foureyes/canonical_confirm/models.npz'


def verify(p):
    for rel,h in (p['source_sha256']|p['input_sha256']).items():assert sha(ROOT/rel)==h,rel


def freeze():
    assert not PROTOCOL.exists()and not OUT.exists()
    manifest=json.loads((PARENT/'inputs.json').read_text());assert len(manifest['banks'])==64
    source=[HERE/n for n in ['pooled_source_study.py','audit_pooled_source_study.py','canonical_pl.py','weak_weight_study.py']]
    source += [RIPPER/'confidence_weight_search.py']
    inputs=[PARENT/'inputs.json',PARENT/'source-audit.json',MODELS,HERE/'canonical_confirm.json',ROOT/'results/research/foureyes/strong_support_study/summary.json']
    inputs += [ROOT/r['path']for r in manifest['banks']]
    p=dict(id='H4ac-fixed-total-observations-majority-versus-pooling',frozen_utc=datetime.now(timezone.utc).isoformat(),
           groups=[list(range(4*g,4*g+4))for g in range(16)],
           scope='Source-only post-hoc mechanism on all64 H4y banks, partitioned into16 disjoint fixed quartets. Same2400 noisy observations/free masks per quartet, including repeated source-row appearances. No new source rows, puzzle rows or inference.',
           arms=['adamw_vote3','canonical_vote3','canonical_pooled','restricted_pooled'],
           primary='Canonical strong-support3-of4 majority is exact in at least15/16 groups AND retains all810 trueedges in every16group.',
           decision='Finite16group target: supported if exact>=15 and missing_true_total==0, otherwise falsified. No population-confidence or fresh-generalization claim.',
           controls='BPAdamW3-of4 vote uses archived separately optimized supports and is first. Pooledcanonical uses same2400 noisyboards once with samefinite256update recipe. Pooledrestrictedcounts uses fixed below-median largest-count-gap rule. Both pooled controls descriptive; no tuned-compute/speed/superiority claim.',
           majority_rule='Each canonical vote is the unchanged H4aa J>0 and J>=median(positiveJ)/4 support; eachgroup retains anedge iff>=3of4votes. Allbanks preserved, no geometry in construction.',
           pooled_fit=dict(raw_lambda=1/30000,updates=256,initial_step=540.,dtype='float64',backtracking='Unchanged canonical_pl.fit,32maxhalvings and carry acceptedstep; no optimum claim.'),
           pooled_rule='Same median-positive-J/4 threshold applied to pooled fit. Do not retune threshold or lambda for2400 rows.',
           gate='Before fitting, eachGPU compares Torch objective/gradient to independent NumPy on one fixed2400rowquartet at zero and two archived coefficient vectors; loss<1e-12,gradient<1e-12. All4gates required.',
           audits='Reconstruct every source concat and vote, no source-row dedup; independent NumPy final loss/gradient/KKT, all256history entries/monotonic objectives, fit/source hashes and all16 graph stats. Failed/ineligiblefits retained and reported, never replaced.',
           budget=dict(gate_phase_wall_seconds=45,fit_phase_wall_seconds=180,max_group_wall_seconds=30,total_gpu_minutes_upper_bound=15),
           source_sha256={str(f.relative_to(ROOT)):sha(f)for f in source},input_sha256={str(f.relative_to(ROOT)):sha(f)for f in inputs})
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL))))


def prepare():
    import numpy as np
    p=json.loads(PROTOCOL.read_text());verify(p);assert not OUT.exists();OUT.mkdir(parents=True)
    old=json.loads((PARENT/'inputs.json').read_text());groups=[];queue=[]
    for g,banks in enumerate(p['groups']):
        arrays={k:[]for k in ['noisy','free','source_row']}
        for b in banks:
            e=old['banks'][b];assert e['bank']==b and sha(ROOT/e['path'])==e['sha256']
            with np.load(ROOT/e['path'],allow_pickle=False)as z:
                for k in arrays:arrays[k].append(z[k].copy())
        arrays={k:np.concatenate(v)for k,v in arrays.items()};arrays['source_bank']=np.repeat(np.array(banks),600)
        path=OUT/'inputs'/f'g{g:02d}.npz';path.parent.mkdir(exist_ok=True);np.savez_compressed(path,**arrays)
        groups.append(dict(group=g,banks=banks,path=str(path.relative_to(ROOT)),sha256=sha(path)))
        queue.append(dict(job_id=f'g{g:02d}',bank=g,status='pending'))
    save(OUT/'inputs.json',dict(protocol_sha256=sha(PROTOCOL),groups=groups));save(OUT/'queue.json',queue)
    print(json.dumps(dict(groups=16,observations_per_group=2400,new_source_rows=0)))


def gate(gpu):
    lock,torch,np,*unused=setup(gpu);p=json.loads(PROTOCOL.read_text());verify(p)
    a=json.loads((OUT/'source-audit.json').read_text());assert a['status']=='validated'and a['protocol_sha256']==sha(PROTOCOL)
    assert a['input_manifest_sha256']==sha(OUT/'inputs.json')
    import canonical_pl as C
    from audit_pooled_source_study import numpy_objective
    e=json.loads((OUT/'inputs.json').read_text())['groups'][gpu];assert sha(ROOT/e['path'])==e['sha256']
    with np.load(ROOT/e['path'],allow_pickle=False)as z:noisy=z['noisy'].copy();free=z['free'].copy()
    with np.load(MODELS,allow_pickle=False)as z:old=z['canonical_weak_coefficients']
    ctx=C.context(torch,torch.as_tensor(noisy.astype(np.int64),device='cuda'),torch.as_tensor(free,device='cuda'),torch.float64)
    results=[]
    for j in [np.zeros(3240),old[4*gpu],old[4*gpu+1]]:
        expected_loss,expected_grad=numpy_objective(j,noisy,free)
        loss,grad=C.loss_gradient(torch.as_tensor(j,device='cuda'),ctx)
        le=abs(float(loss)-expected_loss);ge=float(np.max(np.abs(grad.cpu().numpy()-expected_grad)))
        assert le<1e-12 and ge<1e-12;results.append(dict(loss_error=le,gradient_error=ge))
    save(OUT/f'gate-gpu{gpu}.json',dict(status='passed',gpu=gpu,protocol_sha256=sha(PROTOCOL),cases=results,rows=2400))


def fit(gpu):
    lock,torch,np,*unused=setup(gpu);p=json.loads(PROTOCOL.read_text());verify(p)
    a=json.loads((OUT/'source-audit.json').read_text());assert a['status']=='validated'and a['protocol_sha256']==sha(PROTOCOL)
    assert a['input_manifest_sha256']==sha(OUT/'inputs.json')
    for n in range(4):
        g=json.loads((OUT/f'gate-gpu{n}.json').read_text());assert g['status']=='passed'and g['protocol_sha256']==sha(PROTOCOL)
    import canonical_pl as C
    import confidence_weight_search as queue
    queue.OUT=OUT;entries=json.loads((OUT/'inputs.json').read_text())['groups']
    while(job:=queue.claim(gpu))is not None:
        try:
            start=time.monotonic();group=job['bank'];e=entries[group];assert e['group']==group and sha(ROOT/e['path'])==e['sha256']
            with np.load(ROOT/e['path'],allow_pickle=False)as z:
                sol=torch.as_tensor(z['noisy'].astype(np.int64),device='cuda');free=torch.as_tensor(z['free'],device='cuda')
            j,history=C.fit(torch,sol,free,1/30000,256,540.,torch.float64)
            d=OUT/'fits'/f'g{group:02d}';d.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(d/'model.npz',pooled_coefficients=j.cpu().numpy())
            save(d/'fit.json',history)
            receipt=dict(group=group,gpu=gpu,protocol_sha256=sha(PROTOCOL),input_sha256=e['sha256'],torch=torch.__version__,cuda=torch.version.cuda,
                         wall_seconds=time.monotonic()-start,output_sha256={n:sha(d/n)for n in ['model.npz','fit.json']},finite=bool(torch.isfinite(j).all()),positive=int((j>0).sum()))
            save(d/'receipt.json',receipt)
            assert receipt['wall_seconds']<30 and receipt['finite']
            queue.claim(gpu,job['job_id'],'complete');print(json.dumps(dict(group=group,wall_seconds=receipt['wall_seconds'])),flush=True)
        except BaseException:queue.claim(gpu,job['job_id'],'failed');raise


def launch(phase):
    p=json.loads(PROTOCOL.read_text());verify(p);start=time.monotonic();workers=[];logs=[];samples=[]
    try:
        for gpu in range(4):
            log=(OUT/f'{phase}-gpu{gpu}.log').open('x');logs.append(log)
            workers.append(subprocess.Popen([str(ROOT/'upstream/.venv/bin/python'),str(Path(__file__)),phase,'--gpu',str(gpu)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        while any(w.poll()is None for w in workers):
            assert time.monotonic()-start<p['budget'][phase+'_phase_wall_seconds']and all(w.poll()in[None,0]for w in workers)
            q=json.loads((OUT/'queue.json').read_text());samples.append(dict(elapsed=time.monotonic()-start,complete=sum(x['status']=='complete'for x in q),running=sum(x['status']=='running'for x in q),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True)))
            time.sleep(.5)
        assert all(w.returncode==0 for w in workers)
        result=dict(status='complete',wall_seconds=time.monotonic()-start,samples=samples,returncodes=[w.returncode for w in workers]);save(OUT/(phase+'-launch.json'),result);print(json.dumps({k:v for k,v in result.items()if k!='samples'}))
    finally:
        for w in workers:
            if w.poll()is None:
                os.killpg(w.pid,signal.SIGTERM)
                try:w.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(w.pid,signal.SIGKILL);w.wait()
        for f in logs:f.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','prepare','gate','fit','launch-gate','launch-fit']);ap.add_argument('--gpu',type=int);a=ap.parse_args()
    if a.action in ['gate','fit']:globals()[a.action](a.gpu)
    elif a.action.startswith('launch-'):launch(a.action.split('-')[1])
    else:globals()[a.action]()
