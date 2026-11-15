"""H4af: independent 2400-row banks disjoint from the H4y motivating sources."""
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

OUT=ROOT/'artifacts/research/20261114/foureyes/regularity_fresh'
PUBLIC=ROOT/'results/research/foureyes/regularity_fresh'
PROTOCOL=HERE/'regularity_fresh.json'
PARENT=ROOT/'artifacts/research/20261114/foureyes/canonical_confirm'
DESIGN=ROOT/'artifacts/research/20261114/foureyes/regularity_fresh_design'
ARMS=['adamw','canonical','restricted','regularity']
CONDITION='hot18_weight8'
GROUPS=64
ROWS=2400


def verify(p):
    for rel,digest in (p['source_sha256']|p['input_sha256']).items():assert sha(ROOT/rel)==digest,rel


def bind():
    inherited.OUT=OUT;inherited.PUBLIC=PUBLIC;inherited.PROTOCOL=PROTOCOL
    inherited.CONDITIONS=[CONDITION]
    return inherited


def freeze():
    import numpy as np
    assert not PROTOCOL.exists() and not OUT.exists() and not DESIGN.exists()
    preflight=HERE/'regularity-fresh-preflight.json';gate=json.loads(preflight.read_text());assert gate['status']=='passed'
    for rel,digest in gate['source_sha256'].items():assert sha(ROOT/rel)==digest,rel
    parent=json.loads((HERE/'canonical_confirm.json').read_text())
    previous=json.loads((HERE/'regularity_gap.json').read_text())
    cache=ROOT/'artifacts/research/sudoku_source_cache'/parent['source_sha256_value']
    cm=json.loads((cache/'manifest.json').read_text())
    assert cm['source_sha256']==parent['source_sha256_value']==sha(ROOT/parent['source'])
    for name,digest in cm['files'].items():assert sha(cache/name)==digest
    solutions=np.load(cache/'solutions.npy',mmap_mode='r');given=np.load(cache/'given.npy',mmap_mode='r')
    assert solutions.shape==given.shape and solutions.shape[1]==81
    excluded=np.zeros(len(solutions),bool);evidence={};counts=[]
    for spec in parent['exclusions']:
        path=ROOT/spec['path'];assert sha(path)==spec['sha256'];evidence[spec['path']]=spec['sha256']
        rows=[]
        for entry in json.loads(path.read_text())['inputs']:
            f=ROOT/entry['completion_path'];assert sha(f)==entry['completion_sha256'];evidence[entry['completion_path']]=entry['completion_sha256']
            with np.load(f,allow_pickle=False) as z:rows.append(z['source_row'].copy())
        ids=np.unique(np.concatenate(rows));excluded[ids]=True;counts.append(dict(manifest=spec['path'],unique_ids=len(ids)))
    manifest=json.loads((PARENT/'inputs.json').read_text());h4y=[]
    evidence[str((PARENT/'inputs.json').relative_to(ROOT))]=sha(PARENT/'inputs.json')
    for entry in manifest['banks']:
        f=ROOT/entry['path'];assert sha(f)==entry['sha256'];evidence[entry['path']]=entry['sha256']
        with np.load(f,allow_pickle=False) as z:h4y.append(z['source_row'].copy())
    h4y=np.unique(np.concatenate(h4y));excluded[h4y]=True;eligible=np.flatnonzero(~excluded)
    assert len(eligible)>GROUPS*ROWS
    DESIGN.mkdir(parents=True)
    np.save(DESIGN/'eligible.npy',eligible,allow_pickle=False)
    np.save(DESIGN/'excluded.npy',np.flatnonzero(excluded),allow_pickle=False)
    np.save(DESIGN/'h4y_ids.npy',h4y,allow_pickle=False)
    ledger=dict(status='frozen_before_draws',source=parent['source'],source_sha256=cm['source_sha256'],cache=str(cache.relative_to(ROOT)),
                cache_sha256=cm['files'],total_source_rows=len(solutions),excluded_rows=int(excluded.sum()),eligible_rows=len(eligible),h4y_unique_rows=len(h4y),
                prior_manifest_counts=counts,exclusion_input_sha256=evidence,
                array_sha256={name:sha(DESIGN/name) for name in ['eligible.npy','excluded.npy','h4y_ids.npy']},
                scope='Disjoint from H4y and four declared preceding Ripper manifests only; not globally untouched source identities.')
    save(DESIGN/'ledger.json',ledger)
    source=[HERE/n for n in ['regularity_fresh_study.py','audit_regularity_fresh.py','preflight_regularity_fresh.py',
                            'regularity_gap.py','audit_regularity_gap.py','heterogeneous_noise_study.py','audit_heterogeneous_noise.py',
                            'canonical_pl.py','noisy_pl_long.py','weak_weight_study.py','audit_pooled_source_study.py','pooled_source_study.py']]
    source += [RIPPER/'confidence_weight_search.py',ROOT/'upstream/experiments/sudoku.py']
    inputs=[DESIGN/name for name in ['eligible.npy','excluded.npy','h4y_ids.npy','ledger.json']]
    inputs += [HERE/'regularity_gap.json',HERE/'canonical_confirm.json',preflight,cache/'manifest.json',ROOT/'results/research/foureyes/canonical_confirm/models.npz']
    p=dict(id='H4af-fresh-regularity-confirmation',frozen_utc=datetime.now(timezone.utc).isoformat(),groups=[[g] for g in range(GROUPS)],group_count=GROUPS,rows_per_group=ROWS,
           condition=CONDITION,arms=ARMS,bank_rng_base=202611670000,heatmap_seed_base=202611680000,noise_rng_base=202611690000,augmentation_rng_base=202611700000,
           scope='64 independently sampled2400row sourcebanks without replacement withinbank, crossbank overlap allowed. Draws exclude all H4y motivating sourceIDs plus four frozen earlier Ripper exclusions. Eachbank uses2400distinct rows rather than repeated quartet appearances; same official train corpus, not external or globally untouched sources.',
           sampling='Independent NumPy seeded choice from sorted frozen eligible IDs. Seal all64 sourceID arrays/manifest before reading selected clean labels or generating corruption. No outcomes used to filter or replace banks.',
           source_ledger=str((DESIGN/'ledger.json').relative_to(ROOT)),source_ledger_sha256=sha(DESIGN/'ledger.json'),
           noise='One independently seeded18cellset perbank, sorted and fixed withinbank. Weight8 onhotcells,1elsewhere. Exactly9freecells perboard selected by -log(U)/weight withU=1-rng.random in(0,1],stablecell-index ties. Independentcellwise nonzero digit offsets;clues unchanged. Allfour estimators shareone identical noisybank.',noise_scope=previous['noise_scope'],
           primary='Regularity-selected graphs exact in at least61/64banks ANDzero missing trueedges across all64.',
           decision='Supported if exact>=61 ANDmissing_total==0; otherwise falsified after all64 complete. Numerical/resource failure incomplete. Finite decision, not a95percent population confidence claim.',
           regularity_rule=previous['regularity_rule'],prior=previous['prior'],controls=previous['controls'],diagnostics=previous['diagnostics'],
           adamw=previous['adamw'],canonical=previous['canonical'],inherited_rules=previous['inherited_rules'],gate=previous['gate'],
           budget=dict(preparation_wall_seconds=60,gate_phase_wall_seconds=45,fit_phase_wall_seconds=180,max_group_wall_seconds=30,selection_wall_seconds=30,audit_wall_seconds=180,total_gpu_minutes_upper_bound=15),
           source_sha256={str(f.relative_to(ROOT)):sha(f) for f in source},input_sha256={str(f.relative_to(ROOT)):sha(f) for f in inputs})
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL),eligible_rows=len(eligible),excluded_rows=int(excluded.sum()))))


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
        noisy,cells=inherited.corrupt(np,raw['solutions'],raw['free'],raw['uniforms'],raw['offsets'],hot,CONDITION)
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
