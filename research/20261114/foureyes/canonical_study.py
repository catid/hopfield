"""H4w: fresh-bank convex digit-invariant PL versus count and AdamW controls."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import fcntl
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/research/20261114/foureyes/canonical_study'
PUBLIC=ROOT/'results/research/foureyes/canonical_study'
PROTOCOL=HERE/'canonical_study.json'
ARMS=['adamw','canonical_half','canonical_weak','restricted']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def base_module():
    import noisy_pl_long as base
    base.OUT=OUT;base.PROTOCOL=PROTOCOL
    return base


def freeze():
    import numpy as np
    assert not PROTOCOL.exists() and not OUT.exists()
    gates={}
    for directory in ['canonical_pl_gate','canonical_half_gate']:
        folder=ROOT/'artifacts/research/20261114/foureyes'/directory
        for bank in range(1,5):
            path=folder/f'bank{bank}.json';r=json.loads(path.read_text());assert r['status']=='passed'
            arrays=np.load(folder/f'bank{bank}-coefficients.npz',allow_pickle=False)
            a,b=arrays['cpu64'],arrays['gpu64'];error=np.abs(a-b).max()
            assert np.array_equal(a>0,b>0) and min(a[a>0].min(),b[b>0].min())>4*error
            gates[str(path.relative_to(ROOT))]=sha(path)
            gates[str((folder/f'bank{bank}-coefficients.npz').relative_to(ROOT))]=sha(folder/f'bank{bank}-coefficients.npz')
    parent=json.loads((HERE/'noisy_pl_long.json').read_text())
    names=['id','frozen_utc','source','source_sha256_value','sampling','exclusions','n','banks','adamw','source_sha256','frozen_inputs','search_seed','split','checkpoints','limits']
    p={k:parent[k] for k in names}
    p.update(id='H4w-canonical-convex-PL-positive-support',frozen_utc=datetime.now(timezone.utc).isoformat(),
        hypothesis='CanonicalPL at rawlambda1/6000 with positive-support unitgraph is noninferior within2pp torestrictedcounts, withzero wrongcertifiedstops.',
        arms=ARMS,primary_arm='canonical_half',control_arm='restricted',primary_margin_pp=-2.,
        decision='Supported if pairedtwo-way95lower>=-2pp andzero primarywrongstops; falsified ifupper<-2pp,anyprimarywrongstop,orprimaryfitineligible;otherwiseinconclusive.',
        uncertainty='5000paired two-waybank/input bootstrapdraws,RNG20261115999;64freshsourcebanks,10000sharedpuzzles,onefixedsearchseed.',
        secondary='canonical_weak isdescriptive; AdamWcontrol hasdifferentoptimizer/objective/regularizer andusesoriginalfixedgraphtransform.',
        bank_rng_base=202611300000,noise_rng_base=202611310000,augmentation_rng_base=202611320000,
        updates=256,raw_penalties={'canonical_half':1/6000,'canonical_weak':1/30000},
        canonical_recipe='UnrestrictedrealJ; W=JI,B0; unaugmentedPL exactall9!digit-average onthisparametrization;9lambdaL1;FP64proximalgradient256updates,initialstep540,backtrackinghalves,max32attempts/update,carryacceptedstep. No freshgeometry.',
        graph_rule='Canonicalarms retainJ>0 exactly, useunitweights/B0/tau0. AdamWretainsH4sfirstlargestpositivegaptransform. Restrictedcounts retainH4sbelow-median-countgaprule.',
        original_raw_gate='TenfoldweakerrawSGD failed3/4sourceprecisionchecks; nofreshrawweakSGDexperiment. Newcanonicalparametrization,optimizer,andgraphselection arejointchanges.',
        source_design='Adaptivedesignafter fourpriorN600sourcefixtures. Allsourcegate/thresholddiagnostics retained; nofreshinputs used tochoose recipes.',
        rows_half_open=[390000,400000],chains_by_arm={arm:4 for arm in ARMS},
        calibration='Literalunit1,B0,tau0,ebar0,focus.45,temp.4to.31,anneal524288,check4096,4chains,262144moves/chain;unchanged inheritedsearchloop.',
        budget={'fit_phase_wall_seconds':180,'search_phase_wall_seconds':270,'total_gpu_minutes_upper_bound':30,'max_fit_bank_seconds':30},
        source_gates=gates,fit_failure_policy='Retainnonfinite/ineligiblefits; failprimaryeligibility. Never replaceorretunebanks; donot inventtestaccuracy forineligiblemodels.')
    p['frozen_inputs'].update(gates)
    for name in ['canonical_study.py','audit_canonical_study.py','canonical_pl.py','canonical_pl_gate.py','canonical_half_gate.py','audit_noisy_pl_long.py']:
        p['source_sha256'][str((HERE/name).relative_to(ROOT))]=sha(HERE/name)
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL))))


def verify_frozen(p):
    for name,digest in (p['source_sha256']|p['frozen_inputs']).items():assert sha(ROOT/name)==digest,name


def fit_worker(gpu):
    assert platform.node()=='foureyes'
    lock=(ROOT/f'artifacts/gpu{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    uuid=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    base=base_module();np,torch,CF=base.dependencies(gpu)
    import canonical_pl as C
    p=json.loads(PROTOCOL.read_text());verify_frozen(p)
    manifest=json.loads((OUT/'inputs.json').read_text());assert manifest['protocol_sha256']==sha(PROTOCOL)
    while (job:=base.claim(gpu)) is not None:
        bank=job['bank'];start=time.monotonic()
        try:
            entry=manifest['banks'][bank];path=ROOT/entry['path'];assert sha(path)==entry['sha256']
            z=np.load(path,allow_pickle=False);sol=torch.as_tensor(z['noisy'].astype(np.int64),device='cuda');free=torch.as_tensor(z['free'],device='cuda')
            dest=OUT/'fits'/f'b{bank:02d}';dest.mkdir(parents=True,exist_ok=False)
            draws=torch.rand(32,600,9,generator=torch.Generator().manual_seed(p['augmentation_rng_base']+bank))
            w,b,seconds,history=base.fit_adam(torch,CF,sol,free,draws,p['adamw'])
            assert torch.isfinite(w).all() and torch.isfinite(b).all()
            support,coefficients,meta=base.transform(w,torch)
            torch.save(dict(W=w.cpu(),B=b.cpu()),dest/'adamw.pt')
            arrays={'adamw_coefficients':coefficients.cpu().numpy()};records={'adamw':dict(**meta,loop_s=seconds,loss_history=history)}
            if support is not None:arrays['adamw_support']=support.cpu().numpy()
            for arm,penalty in p['raw_penalties'].items():
                j,fit=C.fit(torch,sol,free,penalty,p['updates'],540.,torch.float64)
                assert torch.isfinite(j).all()
                support=j>0;eligible=bool(support.any())
                arrays[arm+'_coefficients']=j.cpu().numpy();arrays[arm+'_support']=support.cpu().numpy()
                save(dest/(arm+'-fit.json'),fit)
                records[arm]=dict(eligible=eligible,retained=int(support.sum()),loop_s=fit['seconds'],kkt_linf=fit['kkt_linf'],threshold=0.,raw_lambda=penalty)
            ii,jj=np.triu_indices(81,1);counts=(z['noisy'][:,ii]==z['noisy'][:,jj]).sum(0)
            distinct=np.unique(counts);choices=np.flatnonzero(distinct[1:]<=np.median(counts));index=int(choices[np.argmax(np.diff(distinct)[choices])]) if len(choices) else None
            threshold=int((distinct[index]+distinct[index+1])//2) if index is not None else 0
            arrays.update(counts=counts,restricted_support=counts<=threshold);records['restricted']=dict(eligible=True,threshold=threshold)
            np.savez_compressed(dest/'models.npz',**arrays)
            save(dest/'receipt.json',dict(bank=bank,gpu=gpu,protocol_sha256=sha(PROTOCOL),input_sha256=sha(path),arms=records,
                wall_s=time.monotonic()-start,output_sha256={f.name:sha(f) for f in dest.iterdir() if f.is_file()}))
            assert time.monotonic()-start<p['budget']['max_fit_bank_seconds']
            base.claim(gpu,bank,'complete');print(json.dumps(dict(bank=bank,status='complete',wall_s=time.monotonic()-start)),flush=True)
        except BaseException:
            base.claim(gpu,bank,'failed');raise


def map_models():
    import map_noisy_pl_long as mapper
    mapper.OUT=OUT;mapper.PROTOCOL=PROTOCOL
    mapper.main()


def search_worker(gpu,initialize=False):
    import noisy_pl_long_search as search
    search.OUT=OUT;search.PROTOCOL=PROTOCOL
    sys.argv=[str(__file__),'--initialize'] if initialize else [str(__file__),'--gpu',str(gpu)]
    search.main()


def launch(phase):
    p=json.loads(PROTOCOL.read_text());verify_frozen(p)
    if phase=='search':search_worker(None,True)
    limit=p['budget'][phase+'_phase_wall_seconds'];start=time.monotonic();workers=[];logs=[];samples=[]
    try:
        for gpu in range(4):
            log=(OUT/f'{phase}-gpu{gpu}.log').open('x');logs.append(log)
            workers.append(subprocess.Popen([str(ROOT/'upstream/.venv/bin/python'),str(Path(__file__)),phase,'--gpu',str(gpu)],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True))
        while any(w.poll() is None for w in workers):
            assert time.monotonic()-start<limit and all(w.poll() in [None,0] for w in workers)
            q=json.loads((OUT/('fit_queue.json' if phase=='fit' else 'queue.json')).read_text())
            samples.append(dict(elapsed=time.monotonic()-start,complete=sum(r['status']=='complete' for r in q),running=sum(r['status']=='running' for r in q),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True)))
            time.sleep(2)
        assert all(w.returncode==0 for w in workers)
        result=dict(status='complete',phase=phase,wall_seconds=time.monotonic()-start,samples=samples,returncodes=[w.returncode for w in workers])
        save(OUT/(phase+'-launch.json'),result);print(json.dumps({k:v for k,v in result.items() if k!='samples'}))
    finally:
        for w in workers:
            if w.poll() is None:
                os.killpg(w.pid,signal.SIGTERM)
                try:w.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(w.pid,signal.SIGKILL);w.wait()
        for log in logs:log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','prepare','fit','map','search','launch-fit','launch-search']);parser.add_argument('--gpu',type=int);args=parser.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='prepare':
        p=json.loads(PROTOCOL.read_text());verify_frozen(p);base_module().prepare(p)
    elif args.action=='fit':fit_worker(args.gpu)
    elif args.action=='map':map_models()
    elif args.action=='search':search_worker(args.gpu)
    else:launch(args.action.split('-')[1])
