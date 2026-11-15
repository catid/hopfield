"""H4ak: fresh-source confirmation of exact degree-curvature selection."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import os
import signal
import subprocess
import time
from weak_weight_study import ROOT,HERE,RIPPER,sha,save
import heterogeneous_noise_study as inherited
from regularity_intensity_study import corrupt

OUT=ROOT/'artifacts/research/20261114/foureyes/fresh_curvature'
PUBLIC=ROOT/'results/research/foureyes/fresh_curvature/controls'
PROTOCOL=HERE/'fresh_curvature.json'
PARENT=ROOT/'artifacts/research/20261114/foureyes/canonical_confirm'
DESIGN=ROOT/'artifacts/research/20261114/foureyes/fresh_curvature_design'
ARMS=['adamw','canonical','restricted','regularity']
CONDITION='hot18_weight32'
GROUPS=64
ROWS=2400


def verify(p):
    for rel,digest in (p['source_sha256']|p['input_sha256']).items():assert sha(ROOT/rel)==digest,rel


def bind():
    inherited.OUT=OUT;inherited.PUBLIC=PUBLIC;inherited.PROTOCOL=PROTOCOL
    inherited.CONDITIONS=[CONDITION]
    return inherited


def exclusion_arrays():
    import numpy as np
    specs=[('results/research/ripper/fresh_transfer_excluded_ids.npz','excluded'),
           ('results/research/ripper/fresh_transfer_selected_sources.npz','all_source_row'),
           ('artifacts/research/20261114/foureyes/regularity_fresh/source-draws.npz','source_rows')]
    pieces=[];evidence={};counts=[]
    for rel,key in specs:
        path=ROOT/rel
        with np.load(path,allow_pickle=False) as z:ids=np.unique(z[key])
        pieces.append(ids);evidence[rel]=sha(path);counts.append(dict(path=rel,key=key,unique_ids=len(ids)))
    prior=ROOT/'artifacts/research/20261114/foureyes/regularity_fresh_design/excluded.npy'
    pieces.append(np.load(prior,allow_pickle=False));evidence[str(prior.relative_to(ROOT))]=sha(prior)
    counts.append(dict(path=str(prior.relative_to(ROOT)),key=None,unique_ids=len(pieces[-1])))
    assert len(np.union1d(pieces[0],pieces[1]))==2045471
    return np.unique(np.concatenate(pieces)),pieces[2],evidence,counts


def freeze():
    import numpy as np
    assert not PROTOCOL.exists() and not OUT.exists() and not DESIGN.exists()
    gate_path=HERE/'fresh-curvature-preflight.json';gate=json.loads(gate_path.read_text());assert gate['status']=='passed'
    for rel,digest in gate['source_sha256'].items():assert sha(ROOT/rel)==digest
    parent=json.loads((HERE/'canonical_confirm.json').read_text());previous=json.loads((HERE/'regularity_fresh.json').read_text())
    lp=json.loads((HERE/'degree_curvature.json').read_text())
    cache=ROOT/'artifacts/research/sudoku_source_cache'/parent['source_sha256_value'];cm=json.loads((cache/'manifest.json').read_text())
    assert cm['source_sha256']==parent['source_sha256_value']==sha(ROOT/parent['source'])
    for name,digest in cm['files'].items():assert sha(cache/name)==digest
    total=len(np.load(cache/'solutions.npy',mmap_mode='r'))
    excluded,motivating,evidence,counts=exclusion_arrays();eligible=np.setdiff1d(np.arange(total,dtype=np.int64),excluded,assume_unique=True)
    assert len(eligible)>GROUPS*ROWS and excluded.min()>=0 and excluded.max()<total
    DESIGN.mkdir(parents=True)
    for name,value in [('eligible.npy',eligible),('excluded.npy',excluded),('motivating.npy',motivating)]:np.save(DESIGN/name,value,allow_pickle=False)
    ledger=dict(status='frozen_before_draws',source=parent['source'],source_sha256=cm['source_sha256'],cache=str(cache.relative_to(ROOT)),cache_sha256=cm['files'],
                total_source_rows=total,excluded_rows=len(excluded),eligible_rows=len(eligible),motivating_unique_rows=len(motivating),prior_manifest_counts=counts,
                exclusion_input_sha256=evidence,array_sha256={name:sha(DESIGN/name) for name in ['eligible.npy','excluded.npy','motivating.npy']},
                scope='Disjoint from H4af source IDs, its declared exclusions, and latest declared Ripper union2045471. Same official corpus; not globally untouched identities.')
    save(DESIGN/'ledger.json',ledger)
    inputs=[DESIGN/name for name in ['eligible.npy','excluded.npy','motivating.npy','ledger.json']]+[gate_path,cache/'manifest.json',HERE/'degree_curvature.json',HERE/'regularity_fresh.json',HERE/'canonical_confirm.json',ROOT/'results/research/foureyes/canonical_confirm/models.npz']
    p=dict(id='H4ak-fresh-degree-curvature-confirmation',frozen_utc=datetime.now(timezone.utc).isoformat(),groups=[[g] for g in range(GROUPS)],group_count=GROUPS,rows_per_group=ROWS,
           condition=CONDITION,arms=ARMS,bank_rng_base=202611940000,heatmap_seed_base=202611950000,noise_rng_base=202611960000,augmentation_rng_base=202611970000,
           scope='64 independent2400distinct-row banks; disjoint declared exclusions. One weight32 hot18 corruption per bank. Prospective confirmation of H4aj; no new puzzle-solving claim.',
           sampling=previous['sampling'],source_ledger=str((DESIGN/'ledger.json').relative_to(ROOT)),source_ledger_sha256=sha(DESIGN/'ledger.json'),
           noise='18 freshly seeded hot cells per bank, weight32; cold weight1. Exactly9 free digits changed by unchanged exponential-race rule. One shared observation bank for all estimators.',noise_scope=previous['noise_scope'],
           primary='At least61/64 certified integral exact degree-curvature selected graphs.',decision=lp['decision'],
           degrees=lp['degrees'],candidate_degrees=lp['candidate_degrees'],selector=lp['selector'],prior=lp['prior'],objective=lp['objective'],proof=lp['proof'],solver=lp['solver'],
           controls='BP AdamW32 first; canonicalPL256; raw restricted-gap counts; raw regularity; centered regularity. Same observations. Degree20 grid solution diagnostic supplies20. Finite fixed optimizer recipes, not optimum or matched compute claims.',
           regularity_rule=previous['regularity_rule'],adamw=previous['adamw'],canonical=previous['canonical'],inherited_rules=previous['inherited_rules'],gate=previous['gate'],
           budget=dict(preparation_wall_seconds=60,gate_phase_wall_seconds=45,fit_phase_wall_seconds=180,max_group_wall_seconds=30,selection_wall_seconds=30,audit_wall_seconds=240,total_gpu_minutes_upper_bound=15),
           lp_budget=dict(workers=8,new_LPs=2560,aggregate_cpu_seconds=900,wall_seconds=180,lp_wall_seconds=1.,gpu_seconds=0),
           audit_optimization='Eight independent direct per-bank proof workers; exact parity gate on archived eight banks before fresh draws. Same all-proof and mutation checks.',
           source_sha256=gate['source_sha256'],input_sha256={str(f.relative_to(ROOT)):sha(f) for f in inputs})
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL),eligible_rows=len(eligible),excluded_rows=len(excluded))))


def prepare():
    import numpy as np
    p=json.loads(PROTOCOL.read_text());verify(p);assert not OUT.exists();OUT.mkdir(parents=True);start=time.monotonic()
    ledger=json.loads((DESIGN/'ledger.json').read_text());assert sha(DESIGN/'ledger.json')==p['source_ledger_sha256']
    eligible=np.load(DESIGN/'eligible.npy',allow_pickle=False)
    draws=np.stack([np.random.default_rng(p['bank_rng_base']+g).choice(eligible,ROWS,replace=False) for g in range(GROUPS)])
    np.savez_compressed(OUT/'source-draws.npz',source_rows=draws)
    save(OUT/'source-draws.json',dict(protocol_sha256=sha(PROTOCOL),source_ledger_sha256=p['source_ledger_sha256'],
                                   array_sha256=sha(OUT/'source-draws.npz'),bank_rng_base=p['bank_rng_base'],rows_per_bank=ROWS,banks=GROUPS,
                                   sealed_before_materializing_labels=True,unique_source_rows=int(len(np.unique(draws))),
                                   crossbank_repeated_instances=int(draws.size-len(np.unique(draws)))))
    cache=ROOT/ledger['cache']
    for name,digest in ledger['cache_sha256'].items():assert sha(cache/name)==digest
    clean=np.load(cache/'solutions.npy',mmap_mode='r');given=np.load(cache/'given.npy',mmap_mode='r')
    entries=[];jobs=[];heatmaps=[]
    for group,rows in enumerate(draws):
        hot=np.sort(np.random.default_rng(p['heatmap_seed_base']+group).choice(81,18,replace=False));heatmaps.append(hot.tolist())
        rng=np.random.default_rng(p['noise_rng_base']+group)
        raw=dict(solutions=np.asarray(clean[rows]),free=~given[rows],source_row=rows,source_bank=np.full(ROWS,group,np.int64),
                 uniforms=1-rng.random((ROWS,81)),offsets=rng.integers(1,9,(ROWS,81),dtype=np.int64))
        assert (raw['free'].sum(1)>=9).all()
        noisy,cells=corrupt(np,raw['solutions'],raw['free'],raw['uniforms'],raw['offsets'],hot,CONDITION)
        job_id=f'g{group:02d}-{CONDITION}';path=OUT/'inputs'/f'{job_id}.npz';path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,**raw,noisy=noisy,cells=cells)
        entries.append(dict(job_id=job_id,group=group,condition=CONDITION,banks=[group],hot_cells=hot.tolist(),path=str(path.relative_to(ROOT)),sha256=sha(path)))
        jobs.append(dict(job_id=job_id,bank=group,condition=CONDITION,status='pending'))
        assert time.monotonic()-start<p['budget']['preparation_wall_seconds']
    save(OUT/'inputs.json',dict(protocol_sha256=sha(PROTOCOL),hot_cells_by_group=heatmaps,entries=entries,source_draws_sha256=sha(OUT/'source-draws.npz'),source_draws_receipt_sha256=sha(OUT/'source-draws.json')))
    save(OUT/'queue.json',jobs);save(OUT/'preparation.json',dict(wall_seconds=time.monotonic()-start,entries=GROUPS,new_source_draw_instances=GROUPS*ROWS,new_solver_inputs=0))
    print(json.dumps(dict(entries=GROUPS,preparation_wall_seconds=time.monotonic()-start)))

def select_graphs():
    import numpy as np
    from regularity_gap import select
    p=json.loads(PROTOCOL.read_text());verify(p);start=time.monotonic()
    source=json.loads((OUT/'source-audit.json').read_text())
    assert source['status']=='validated' and source['protocol_sha256']==sha(PROTOCOL) and source['input_manifest_sha256']==sha(OUT/'inputs.json')
    manifest=json.loads((OUT/'inputs.json').read_text());queue=json.loads((OUT/'queue.json').read_text())
    assert len(queue)==GROUPS and all(j['status']=='complete' for j in queue)
    destination=OUT/'regularity';destination.mkdir(exist_ok=False);rows=[];supports=[]
    for e in manifest['entries']:
        d=OUT/'fits'/e['job_id'];receipt=json.loads((d/'receipt.json').read_text())
        assert receipt['protocol_sha256']==sha(PROTOCOL) and receipt['input_sha256']==e['sha256']
        assert sha(d/'models.npz')==receipt['output_sha256']['models.npz']
        with np.load(d/'models.npz',allow_pickle=False) as z:counts=z['counts'].copy()
        support,record=select(counts)
        record.update(job_id=e['job_id'],group=e['group'],protocol_sha256=sha(PROTOCOL),models_sha256=sha(d/'models.npz'))
        path=destination/(e['job_id']+'.json');save(path,record)
        supports.append(support);rows.append(dict(group=e['group'],job_id=e['job_id'],record_sha256=sha(path)))
        assert time.monotonic()-start<p['budget']['selection_wall_seconds']
    np.savez_compressed(destination/'supports.npz',supports=np.stack(supports),groups=np.arange(GROUPS))
    save(destination/'receipt.json',dict(status='complete',protocol_sha256=sha(PROTOCOL),rows=rows,supports_sha256=sha(destination/'supports.npz'),wall_seconds=time.monotonic()-start,selector_sha256=sha(HERE/'regularity_gap.py')))
    print(json.dumps(dict(status='complete',groups=GROUPS,selection_wall_seconds=time.monotonic()-start)))


def launch(phase):
    p=json.loads(PROTOCOL.read_text());verify(p);start=time.monotonic();cpu_start=inherited.cpu_seconds()
    workers=[];logs=[];samples=[];status='failed';failure=None
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
        assert all(w.returncode==0 for w in workers);status='complete'
    except BaseException as error:failure=repr(error);raise
    finally:
        for w in workers:
            if w.poll() is None:
                os.killpg(w.pid,signal.SIGTERM)
                try:w.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(w.pid,signal.SIGKILL);w.wait()
        for f in logs:f.close()
        result=dict(status=status,failure=failure,wall_seconds=time.monotonic()-start,aggregate_cpu_seconds=inherited.cpu_seconds()-cpu_start,samples=samples,returncodes=[w.returncode for w in workers])
        save(OUT/f'{phase}-launch.json',result);print(json.dumps({k:v for k,v in result.items() if k!='samples'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','prepare','gate','fit','select','launch-gate','launch-fit']);parser.add_argument('--gpu',type=int);args=parser.parse_args()
    if args.action in ['gate','fit']:getattr(bind(),args.action)(args.gpu)
    elif args.action.startswith('launch-'):launch(args.action.split('-')[1])
    elif args.action=='select':select_graphs()
    else:globals()[args.action]()
