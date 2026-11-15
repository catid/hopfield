"""H4ag: paired corruption intensity on the same 64 H4af clean banks."""
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

OUT=ROOT/'artifacts/research/20261114/foureyes/regularity_intensity'
PUBLIC=ROOT/'results/research/foureyes/regularity_intensity'
PROTOCOL=HERE/'regularity_intensity.json'
PARENT=ROOT/'artifacts/research/20261114/foureyes/regularity_fresh'
ARMS=['adamw','canonical','restricted','regularity']
CONDITIONS=['hot18_weight8','hot18_weight32']
GROUPS=64
ROWS=2400


def verify(p):
    for rel,digest in (p['source_sha256']|p['input_sha256']).items():assert sha(ROOT/rel)==digest,rel


def bind():
    inherited.OUT=OUT;inherited.PUBLIC=PUBLIC;inherited.PROTOCOL=PROTOCOL
    inherited.CONDITIONS=CONDITIONS
    return inherited


def corrupt(np, clean, free, uniforms, offsets, hot_cells, condition):
    assert condition in CONDITIONS
    weights=np.ones(81,np.float64)
    weights[hot_cells]={'hot18_weight8':8.,'hot18_weight32':32.}[condition]
    assert (uniforms>0).all() and (uniforms<=1).all() and (free.sum(1)>=9).all()
    priority=np.where(free,-np.log(uniforms)/weights,np.inf)
    cells=np.argsort(priority,axis=1,kind='stable')[:,:9]
    noisy=clean.copy()
    values=(np.take_along_axis(clean,cells,1)+np.take_along_axis(offsets,cells,1))%9
    np.put_along_axis(noisy,cells,values,1)
    return noisy,cells


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists()
    preflight=HERE/'regularity-intensity-preflight.json';gate=json.loads(preflight.read_text());assert gate['status']=='passed'
    for rel,digest in gate['source_sha256'].items():assert sha(ROOT/rel)==digest,rel
    previous=json.loads((HERE/'regularity_fresh.json').read_text())
    manifest=json.loads((PARENT/'inputs.json').read_text());assert len(manifest['entries'])==64
    source=[HERE/n for n in ['regularity_intensity_study.py','audit_regularity_intensity.py','preflight_regularity_intensity.py',
                            'regularity_gap.py','audit_regularity_gap.py','heterogeneous_noise_study.py','audit_heterogeneous_noise.py',
                            'canonical_pl.py','noisy_pl_long.py','weak_weight_study.py','audit_pooled_source_study.py','pooled_source_study.py']]
    source += [RIPPER/'confidence_weight_search.py',ROOT/'upstream/experiments/sudoku.py']
    inputs=[HERE/'regularity_fresh.json',preflight,PARENT/'inputs.json',PARENT/'source-audit.json',
            PARENT/'source-draws.npz',ROOT/'results/research/foureyes/regularity_fresh/source-identities.npz',
            ROOT/'results/research/foureyes/canonical_confirm/models.npz']
    inputs += [ROOT/e['path'] for e in manifest['entries']]
    p=dict(id='H4ag-paired-hotspot-intensity',frozen_utc=datetime.now(timezone.utc).isoformat(),groups=[[g] for g in range(GROUPS)],
           group_count=GROUPS,rows_per_group=ROWS,conditions=CONDITIONS,arms=ARMS,
           heatmap_seed_base=202611710000,noise_rng_base=202611720000,augmentation_rng_base=202611730000,
           scope='Same64 H4af clean2400row banks; new heatmaps and paired corruption draws. No new source identities or puzzle inputs. Finite paired randomization, not fresh-source confirmation.',
           noise='Perbank independently seeded18hotcells. Shared U=1-rng.random and nonzero digit offsets; exactly9 free cells by exponential race -log(U)/weight; stable increasing cell ties. Hot weights8 versus32, cold1. Clues unchanged. Same augmentation seed across arms.',
           primary='Weight32 regularity at least61/64 exact graphs ANDzero missing trueedges.',
           decision='Supported if exact>=61 ANDmissing_total==0; otherwise falsified after all128 jobs and audits complete. Resource/numerical failures incomplete.',
           attribution='Intensity-specific failure only if paired weight8 meets the same criterion and weight32 fails. Otherwise report both endpoint decisions and retain control failure.',
           regularity_rule=previous['regularity_rule'],prior=previous['prior'],controls=previous['controls'],diagnostics=previous['diagnostics'],
           adamw=previous['adamw'],canonical=previous['canonical'],inherited_rules=previous['inherited_rules'],
           gate='All4 GPUs, both conditions: independent rawPL17row and canonicalPL2400row tests at three vectors; same H4af tolerances. Independent literal corruption replay before fits.',
           budget=dict(preparation_wall_seconds=60,gate_phase_wall_seconds=45,fit_phase_wall_seconds=180,max_group_wall_seconds=30,selection_wall_seconds=30,audit_wall_seconds=180,total_gpu_minutes_upper_bound=15),
           source_sha256={str(f.relative_to(ROOT)):sha(f) for f in source},input_sha256={str(f.relative_to(ROOT)):sha(f) for f in inputs})
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL))))


def prepare():
    import numpy as np
    p=json.loads(PROTOCOL.read_text());verify(p);assert not OUT.exists();OUT.mkdir(parents=True);start=time.monotonic()
    old=json.loads((PARENT/'inputs.json').read_text());entries=[];jobs=[];heatmaps=[]
    for group,entry in enumerate(old['entries']):
        assert entry['group']==group and sha(ROOT/entry['path'])==entry['sha256']
        with np.load(ROOT/entry['path'],allow_pickle=False) as z:
            raw={key:z[key].copy() for key in ['solutions','free','source_row','source_bank']}
        hot=np.sort(np.random.default_rng(p['heatmap_seed_base']+group).choice(81,18,replace=False));heatmaps.append(hot.tolist())
        rng=np.random.default_rng(p['noise_rng_base']+group)
        raw.update(uniforms=1-rng.random((ROWS,81)),offsets=rng.integers(1,9,(ROWS,81),dtype=np.int64))
        for condition in CONDITIONS:
            noisy,cells=corrupt(np,raw['solutions'],raw['free'],raw['uniforms'],raw['offsets'],hot,condition)
            job_id=f'g{group:02d}-{condition}';path=OUT/'inputs'/f'{job_id}.npz';path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,**raw,noisy=noisy,cells=cells)
            entries.append(dict(job_id=job_id,group=group,condition=condition,banks=[group],hot_cells=hot.tolist(),path=str(path.relative_to(ROOT)),sha256=sha(path)))
            jobs.append(dict(job_id=job_id,bank=group,condition=condition,status='pending'))
        assert time.monotonic()-start<p['budget']['preparation_wall_seconds']
    save(OUT/'inputs.json',dict(protocol_sha256=sha(PROTOCOL),hot_cells_by_group=heatmaps,entries=entries))
    save(OUT/'queue.json',jobs);save(OUT/'preparation.json',dict(wall_seconds=time.monotonic()-start,entries=len(entries),new_source_draw_instances=0,new_solver_inputs=0))
    print(json.dumps(dict(entries=len(entries),preparation_wall_seconds=time.monotonic()-start)))


def select_graphs():
    import numpy as np
    from regularity_gap import select
    p=json.loads(PROTOCOL.read_text());verify(p);start=time.monotonic()
    source=json.loads((OUT/'source-audit.json').read_text())
    assert source['status']=='validated' and source['protocol_sha256']==sha(PROTOCOL) and source['input_manifest_sha256']==sha(OUT/'inputs.json')
    manifest=json.loads((OUT/'inputs.json').read_text());queue=json.loads((OUT/'queue.json').read_text())
    assert len(queue)==GROUPS*len(CONDITIONS) and all(j['status']=='complete' for j in queue)
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
    np.savez_compressed(destination/'supports.npz',supports=np.stack(supports),groups=np.array([e['group'] for e in manifest['entries']]),conditions=np.array([e['condition'] for e in manifest['entries']]))
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
