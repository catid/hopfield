"""H4ad: same corruption count, paired uniform and fixed cell propensity."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import resource
import signal
import subprocess
import sys
import time
from weak_weight_study import ROOT, HERE, RIPPER, sha, save, setup

OUT = ROOT / 'artifacts/research/20261114/foureyes/heterogeneous_noise'
PUBLIC = ROOT / 'results/research/foureyes/heterogeneous_noise'
PROTOCOL = HERE / 'heterogeneous_noise.json'
PARENT = ROOT / 'artifacts/research/20261114/foureyes/canonical_confirm'
CONDITIONS = ['uniform', 'hot18_weight8']
ARMS = ['adamw', 'canonical', 'restricted']


def verify(p):
    for rel, digest in (p['source_sha256'] | p['input_sha256']).items():
        assert sha(ROOT / rel) == digest, rel


def corrupt(np, clean, free, uniforms, offsets, hot_cells, condition):
    assert condition in CONDITIONS
    weights = np.ones(81, np.float64)
    if condition == 'hot18_weight8':
        weights[hot_cells] = 8
    assert (uniforms > 0).all() and (uniforms <= 1).all()
    priority = np.where(free, -np.log(uniforms) / weights, np.inf)
    # Stable sorting makes exact race ties use increasing cell index.
    cells = np.argsort(priority, axis=1, kind='stable')[:, :9]
    noisy = clean.copy()
    values = (np.take_along_axis(clean, cells, 1) + np.take_along_axis(offsets, cells, 1)) % 9
    np.put_along_axis(noisy, cells, values, 1)
    return noisy, cells


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists()
    preflight = HERE / 'heterogeneous-noise-preflight.json'
    check = json.loads(preflight.read_text())
    assert check['status'] == 'passed'
    for rel, digest in check['source_sha256'].items():
        assert sha(ROOT / rel) == digest, rel
    parent = json.loads((HERE / 'canonical_confirm.json').read_text())
    manifest = json.loads((PARENT / 'inputs.json').read_text())
    sources = [HERE / n for n in ['heterogeneous_noise_study.py', 'audit_heterogeneous_noise.py',
                'preflight_heterogeneous_noise.py', 'canonical_pl.py', 'noisy_pl_long.py',
                'weak_weight_study.py', 'audit_pooled_source_study.py', 'pooled_source_study.py']]
    sources += [RIPPER / 'confidence_weight_search.py', ROOT / 'upstream/experiments/sudoku.py']
    inputs = [PARENT / 'inputs.json', PARENT / 'source-audit.json', HERE / 'canonical_confirm.json',
              ROOT / 'results/research/foureyes/canonical_confirm/models.npz', preflight]
    inputs += [ROOT / e['path'] for e in manifest['banks']]
    p = dict(id='H4ad-heterogeneous-cell-corruption', frozen_utc=datetime.now(timezone.utc).isoformat(),
             groups=[list(range(4*g, 4*g+4)) for g in range(16)], conditions=CONDITIONS, arms=ARMS,
             hotspot_seed=202611600000, noise_rng_base=202611610000, augmentation_rng_base=202611620000,
             noise='One 18-cell set drawn uniformly once, sorted; fixed throughout. Exactly9 free cells per row selected by exponential races, -log(U)/weight, stable cell-index ties; U=1-rng.random in(0,1]. Weights8 on hot set versus1 elsewhere; uniform arm all1. Paired U and per-cell uniform nonzero digit offsets across conditions. Clues unchanged.',
             scope='Same64H4y clean banks in16fixed ordered quartets,2400 row appearances per group including repeats. New paired corruption draws; no new source rows, puzzle inputs or inference. One fixed heat map, not average over locations.',
             primary='Heterogeneous restricted-gap counts exact in at least15/16groups AND zero missing true edges across all16.',
             decision='Supported if exact>=15 and missing_total==0; otherwise falsified once all32fits/source audits complete. Numerical or resource failure leaves study incomplete; no replacement or filtering.',
             controls='BPAdamW first, pooled32updates and inherited transform; canonical256 and restrictedcounts for eachcondition. Different training work/objectives; descriptive controls, no optimizer-superiority claim.',
             attribution='A heterogeneous-only failure is attributed to this intervention only if paired uniform control succeeds. If both fail, state this limitation. Finite previously examined source banks, no population confidence claim.',
             diagnostics='Postfit only: per-cell free/corruption rates; hot-hot/hot-cold/cold-cold missing/extra counts; true/nonpeer count ranges and whether any scalar threshold separates them. No geometry enters corruption, training or graph selection.',
             adamw=parent['adamw'], canonical=dict(raw_lambda=1/30000,updates=256,initial_step=540.,dtype='float64'),
             rules=dict(canonical='J>0 and J>=median(positiveJ)/4',restricted='First largest adjacent distinct-count gap with upper endpoint<=median(counts); floored midpoint, retain counts<=cut; if no gap cut0.',adamw='Unchanged noisy_pl_long.transform, diagonal-minus-offdiagonal means clamped at0, first largest positive gap.'),
             gate='All4GPUs, bothconditions: full2400row canonical loss/gradient at zero and2archived vectors vs independent NumPy errors<1e-12. Unrestricted rawPL on17actual rows at deterministic random W/B: FP32 upstream loss error<2e-6 and dW/dB maxerror<2e-7 vs independent FP64 NumPy.',
             budget=dict(gate_phase_wall_seconds=45,fit_phase_wall_seconds=180,max_group_wall_seconds=30,total_gpu_minutes_upper_bound=15),
             source_sha256={str(f.relative_to(ROOT)):sha(f) for f in sources},
             input_sha256={str(f.relative_to(ROOT)):sha(f) for f in inputs})
    save(PROTOCOL,p)
    print(json.dumps(dict(protocol_sha256=sha(PROTOCOL))))


def prepare():
    import numpy as np
    p=json.loads(PROTOCOL.read_text());verify(p)
    assert not OUT.exists();OUT.mkdir(parents=True)
    start=time.monotonic()
    hot=np.sort(np.random.default_rng(p['hotspot_seed']).choice(81,18,replace=False))
    old=json.loads((PARENT/'inputs.json').read_text());entries=[];jobs=[]
    for group,banks in enumerate(p['groups']):
        raw={k:[] for k in ['solutions','free','source_row','source_bank','uniforms','offsets']}
        for bank in banks:
            e=old['banks'][bank];assert e['bank']==bank and sha(ROOT/e['path'])==e['sha256']
            with np.load(ROOT/e['path'],allow_pickle=False) as z:
                for key in ['solutions','free','source_row']:raw[key].append(z[key].copy())
            rng=np.random.default_rng(p['noise_rng_base']+bank)
            raw['source_bank'].append(np.full(600,bank,np.int64))
            raw['uniforms'].append(1-rng.random((600,81)))
            raw['offsets'].append(rng.integers(1,9,(600,81),dtype=np.int64))
        raw={key:np.concatenate(values) for key,values in raw.items()}
        assert raw['solutions'].shape==raw['free'].shape==(2400,81) and (raw['free'].sum(1)>=9).all()
        for condition in CONDITIONS:
            noisy,cells=corrupt(np,raw['solutions'],raw['free'],raw['uniforms'],raw['offsets'],hot,condition)
            job_id=f'g{group:02d}-{condition}'
            path=OUT/'inputs'/f'{job_id}.npz';path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,**raw,noisy=noisy,cells=cells)
            entries.append(dict(job_id=job_id,group=group,condition=condition,banks=banks,path=str(path.relative_to(ROOT)),sha256=sha(path)))
            jobs.append(dict(job_id=job_id,bank=group,condition=condition,status='pending'))
    save(OUT/'inputs.json',dict(protocol_sha256=sha(PROTOCOL),hot_cells=hot.tolist(),entries=entries))
    save(OUT/'queue.json',jobs)
    save(OUT/'preparation.json',dict(wall_seconds=time.monotonic()-start,entries=32,new_source_rows=0))
    print(json.dumps(dict(entries=32,groups=16,preparation_wall_seconds=time.monotonic()-start)))


def validated_sources(p):
    verify(p)
    a=json.loads((OUT/'source-audit.json').read_text())
    assert a['status']=='validated' and a['protocol_sha256']==sha(PROTOCOL)
    assert a['input_manifest_sha256']==sha(OUT/'inputs.json')
    return json.loads((OUT/'inputs.json').read_text())


def gate(gpu):
    lock,torch,np,*_=setup(gpu)
    import canonical_pl as C
    from audit_pooled_source_study import numpy_objective
    from audit_heterogeneous_noise import numpy_raw_loss_gradient
    import noisy_pl_long as base
    _,_,CF=base.dependencies()
    p=json.loads(PROTOCOL.read_text());manifest=validated_sources(p)
    with np.load(ROOT/'results/research/foureyes/canonical_confirm/models.npz',allow_pickle=False) as z:
        old=z['canonical_weak_coefficients'].copy()
    cases=[]
    for condition in CONDITIONS:
        e=next(e for e in manifest['entries'] if e['group']==gpu and e['condition']==condition)
        assert sha(ROOT/e['path'])==e['sha256']
        with np.load(ROOT/e['path'],allow_pickle=False) as z:noisy=z['noisy'].copy();free=z['free'].copy()
        sol=torch.as_tensor(noisy.astype(np.int64),device='cuda');fr=torch.as_tensor(free,device='cuda')
        ctx=C.context(torch,sol,fr,torch.float64);canonical=[]
        for j in [np.zeros(3240,np.float64),old[4*gpu],old[4*gpu+1]]:
            expected_loss,expected_grad=numpy_objective(j,noisy,free)
            loss,grad=C.loss_gradient(torch.as_tensor(j,device='cuda'),ctx)
            le=abs(float(loss)-expected_loss);ge=float(abs(grad.cpu().numpy()-expected_grad).max())
            assert le<1e-12 and ge<1e-12
            canonical.append(dict(loss_error=le,gradient_error=ge))
        rng=np.random.default_rng(202611630000+gpu)
        w=rng.normal(0,.03,(3240,9,9)).astype(np.float32);b=rng.normal(0,.03,(81,9)).astype(np.float32)
        expected,dw,db=numpy_raw_loss_gradient(w.astype(np.float64),b.astype(np.float64),noisy[:17],free[:17])
        wt=torch.tensor(w,device='cuda',requires_grad=True);bt=torch.tensor(b,device='cuda',requires_grad=True)
        loss=CF.pl_loss(CF.full_S(wt,CF.pair_index('cuda')),bt,sol[:17],fr[:17]);loss.backward()
        le=abs(float(loss.detach())-expected);we=float(abs(wt.grad.cpu().numpy()-dw).max());be=float(abs(bt.grad.cpu().numpy()-db).max())
        assert le<2e-6 and max(we,be)<2e-7
        cases.append(dict(condition=condition,input_sha256=e['sha256'],canonical=canonical,raw=dict(loss_error=le,w_gradient_error=we,b_gradient_error=be,rows=17)))
    save(OUT/f'gate-gpu{gpu}.json',dict(status='passed',gpu=gpu,protocol_sha256=sha(PROTOCOL),cases=cases))


def fit(gpu):
    lock,torch,np,*_=setup(gpu)
    import canonical_pl as C
    import noisy_pl_long as base
    import confidence_weight_search as queue
    _,_,CF=base.dependencies();queue.OUT=OUT
    p=json.loads(PROTOCOL.read_text());manifest=validated_sources(p)
    for n in range(4):
        record=json.loads((OUT/f'gate-gpu{n}.json').read_text())
        assert record['status']=='passed' and record['protocol_sha256']==sha(PROTOCOL)
    while (job:=queue.claim(gpu)) is not None:
        try:
            start=time.monotonic();group=job['bank'];condition=job['condition']
            e=next(e for e in manifest['entries'] if e['job_id']==job['job_id'])
            assert e['group']==group and e['condition']==condition and sha(ROOT/e['path'])==e['sha256']
            with np.load(ROOT/e['path'],allow_pickle=False) as z:noisy=z['noisy'].copy();free=z['free'].copy()
            sol=torch.as_tensor(noisy.astype(np.int64),device='cuda');fr=torch.as_tensor(free,device='cuda')
            d=OUT/'fits'/job['job_id'];d.mkdir(parents=True,exist_ok=False)
            draws=torch.rand(32,2400,9,generator=torch.Generator().manual_seed(p['augmentation_rng_base']+group))
            w,b,seconds,history=base.fit_adam(torch,CF,sol,fr,draws,p['adamw'])
            assert torch.isfinite(w).all() and torch.isfinite(b).all()
            support,coeff,meta=base.transform(w,torch)
            if support is None:support=torch.zeros(3240,dtype=torch.bool,device='cuda')
            torch.save(dict(W=w.cpu(),B=b.cpu()),d/'adamw.pt')
            j,record=C.fit(torch,sol,fr,1/30000,256,540.,torch.float64)
            assert torch.isfinite(j).all()
            values=j.cpu().numpy();positive=values[values>0]
            cut=float(np.median(positive)/4) if len(positive) else None
            strong=(values>0)&(values>=cut) if cut is not None else np.zeros(3240,bool)
            ii,jj=np.triu_indices(81,1);counts=(noisy[:,ii]==noisy[:,jj]).sum(0)
            distinct=np.unique(counts);choices=np.flatnonzero(distinct[1:]<=np.median(counts))
            index=int(choices[np.argmax(np.diff(distinct)[choices])]) if len(choices) else None
            threshold=int((distinct[index]+distinct[index+1])//2) if index is not None else 0
            np.savez_compressed(d/'models.npz',adamw_coefficients=coeff.cpu().numpy(),adamw_support=support.cpu().numpy(),
                                canonical_coefficients=values,canonical_support=strong,counts=counts,restricted_support=counts<=threshold)
            save(d/'canonical-fit.json',record)
            receipt=dict(group=group,condition=condition,gpu=gpu,protocol_sha256=sha(PROTOCOL),input_sha256=e['sha256'],
                         adamw=dict(**meta,loop_seconds=seconds,history=history,augmentation_seed=p['augmentation_rng_base']+group),
                         canonical_threshold=cut,canonical_eligible=cut is not None,restricted_threshold=threshold,
                         wall_seconds=time.monotonic()-start,output_sha256={n:sha(d/n) for n in ['adamw.pt','models.npz','canonical-fit.json']},
                         torch=torch.__version__,cuda=torch.version.cuda)
            save(d/'receipt.json',receipt)
            assert receipt['wall_seconds']<p['budget']['max_group_wall_seconds']
            queue.claim(gpu,job['job_id'],'complete')
            print(json.dumps(dict(job_id=job['job_id'],wall_seconds=receipt['wall_seconds'])),flush=True)
        except BaseException:
            queue.claim(gpu,job['job_id'],'failed');raise


def cpu_seconds():
    return sum(r.ru_utime+r.ru_stime for r in [resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN)])


def launch(phase):
    p=json.loads(PROTOCOL.read_text());verify(p)
    start=time.monotonic();cpu_start=cpu_seconds();workers=[];logs=[];samples=[];status='failed';failure=None
    try:
        for gpu in range(4):
            log=(OUT/f'{phase}-gpu{gpu}.log').open('x');logs.append(log)
            workers.append(subprocess.Popen([str(ROOT/'upstream/.venv/bin/python'),str(Path(__file__)),phase,'--gpu',str(gpu)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        while any(w.poll() is None for w in workers):
            assert time.monotonic()-start<p['budget'][phase+'_phase_wall_seconds'],'phase_wall_budget'
            assert all(w.poll() in [None,0] for w in workers),'worker_failure'
            q=json.loads((OUT/'queue.json').read_text())
            samples.append(dict(elapsed=time.monotonic()-start,complete=sum(x['status']=='complete' for x in q),running=sum(x['status']=='running' for x in q),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True)))
            time.sleep(.5)
        assert all(w.returncode==0 for w in workers)
        status='complete'
    except BaseException as error:
        failure=repr(error);raise
    finally:
        for w in workers:
            if w.poll() is None:
                os.killpg(w.pid,signal.SIGTERM)
                try:w.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(w.pid,signal.SIGKILL);w.wait()
        for f in logs:f.close()
        result=dict(status=status,failure=failure,wall_seconds=time.monotonic()-start,aggregate_cpu_seconds=cpu_seconds()-cpu_start,samples=samples,returncodes=[w.returncode for w in workers])
        save(OUT/f'{phase}-launch.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='samples'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','prepare','gate','fit','launch-gate','launch-fit']);parser.add_argument('--gpu',type=int)
    args=parser.parse_args()
    if args.action in ['gate','fit']:globals()[args.action](args.gpu)
    elif args.action.startswith('launch-'):launch(args.action.split('-')[1])
    else:globals()[args.action]()
