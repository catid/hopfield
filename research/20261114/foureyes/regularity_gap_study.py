"""H4ae: new cell profiles, unchanged fits, regularity-selected thresholds."""
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

OUT=ROOT/'artifacts/research/20261114/foureyes/regularity_gap'
PUBLIC=ROOT/'results/research/foureyes/regularity_gap'
PROTOCOL=HERE/'regularity_gap.json'
PARENT=ROOT/'artifacts/research/20261114/foureyes/canonical_confirm'
ARMS=['adamw','canonical','restricted','regularity']
CONDITION='hot18_weight8'


def verify(p):
    for rel,digest in (p['source_sha256']|p['input_sha256']).items():assert sha(ROOT/rel)==digest,rel


def bind():
    inherited.OUT=OUT;inherited.PUBLIC=PUBLIC;inherited.PROTOCOL=PROTOCOL
    inherited.CONDITIONS=[CONDITION]
    return inherited


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists()
    preflight=HERE/'regularity-gap-preflight.json';gate=json.loads(preflight.read_text());assert gate['status']=='passed'
    for rel,digest in gate['source_sha256'].items():assert sha(ROOT/rel)==digest,rel
    old=json.loads((HERE/'heterogeneous_noise.json').read_text());manifest=json.loads((PARENT/'inputs.json').read_text())
    source=[HERE/n for n in ['regularity_gap_study.py','regularity_gap.py','audit_regularity_gap.py','preflight_regularity_gap.py',
                            'heterogeneous_noise_study.py','audit_heterogeneous_noise.py','canonical_pl.py','noisy_pl_long.py',
                            'weak_weight_study.py','audit_pooled_source_study.py','pooled_source_study.py']]
    source += [RIPPER/'confidence_weight_search.py',ROOT/'upstream/experiments/sudoku.py']
    inputs=[PARENT/'inputs.json',PARENT/'source-audit.json',HERE/'canonical_confirm.json',HERE/'heterogeneous_noise.json',
            ROOT/'results/research/foureyes/canonical_confirm/models.npz',preflight]
    inputs += [ROOT/e['path'] for e in manifest['banks']]
    p=dict(id='H4ae-regularity-gap-new-cell-profiles',frozen_utc=datetime.now(timezone.utc).isoformat(),
           groups=[list(range(4*g,4*g+4)) for g in range(16)],condition=CONDITION,arms=ARMS,
           heatmap_seed_base=202611640000,noise_rng_base=202611650000,augmentation_rng_base=202611660000,
           scope='Same64H4y clean sourcebanks in16orderedquartets,2400board appearances/group including repeats. One new independently seeded18cellheatmap/group and newcorruptions. Finite randomization,notfreshsource. No newsource/solverinputs.',
           noise='One independently seeded18cellset per quartet, sorted and fixed within quartet. Weight8 onhotcells,1elsewhere. Exactly9freecells perboard selected by -log(U)/weight withU=1-rng.random in(0,1],stablecell-index ties. Independentcellwise nonzero digit offsets;clues unchanged. Allfour estimators shareone identical noisybank perquartet.',
           noise_scope='Only hot18_weight8 is generated. Allfour estimators see exactly the same new observations; no fresh uniform-noise arm.',
           primary='Regularity-selected graph exact in at least15/16groups AND zero missing true edges across all16.',
           decision='Supported if exact>=15 and missing_total==0, otherwise falsified after all16fit/selection/audit jobs complete. Numerical/resource failure remains incomplete; no filtering, replacement or threshold-window changes.',
           regularity_rule='Every distinct observed equality count<=median(counts) yields candidate edges count<=cut; retain all nonempty candidates. Minimize exact integer81*sum(degree^2)-(sumdegree)^2. Ties fewer retainededges,then lowercut. No minimumedge/degree guard. No supplieddegree20 orcardinality810.',
           prior='Equal degree across81cells is supplied structural information; no row/column/box geometry enters selection. The degree value is not supplied.',
           controls='BPAdamW32 first, canonical256 strong threshold and original restrictedlargestgap on identical new observations. Differenttraining work/objectives; descriptive, nooptimizer-superiorityclaim.',
           diagnostics='Record everycut,edgecount,degreemin/max,sum/squares,dispersion andminimumtiecount. Geometry onlyafterselection: separate no separatingcountthreshold from wrongregularitychoice; retain sparse regular failures.',
           adamw=old['adamw'],canonical=old['canonical'],inherited_rules=old['rules'],
           gate='Same independent rawPL and canonicalPL checks asH4ad on all4GPUs, eachone2400rowgroup, onlyhotcondition. Canonical3vectors errors<1e-12; unrestricted17rowFP32loss<2e-6,dW/dB<2e-7.',
           budget=dict(gate_phase_wall_seconds=45,fit_phase_wall_seconds=180,max_group_wall_seconds=30,selection_wall_seconds=30,total_gpu_minutes_upper_bound=15),
           source_sha256={str(f.relative_to(ROOT)):sha(f) for f in source},input_sha256={str(f.relative_to(ROOT)):sha(f) for f in inputs})
    save(PROTOCOL,p);print(json.dumps(dict(protocol_sha256=sha(PROTOCOL))))


def prepare():
    import numpy as np
    p=json.loads(PROTOCOL.read_text());verify(p);assert not OUT.exists();OUT.mkdir(parents=True)
    start=time.monotonic();old=json.loads((PARENT/'inputs.json').read_text());entries=[];jobs=[];heatmaps=[]
    for group,banks in enumerate(p['groups']):
        hot=np.sort(np.random.default_rng(p['heatmap_seed_base']+group).choice(81,18,replace=False));heatmaps.append(hot.tolist())
        raw={k:[] for k in ['solutions','free','source_row','source_bank','uniforms','offsets']}
        for bank in banks:
            e=old['banks'][bank];assert e['bank']==bank and sha(ROOT/e['path'])==e['sha256']
            with np.load(ROOT/e['path'],allow_pickle=False) as z:
                for key in ['solutions','free','source_row']:raw[key].append(z[key].copy())
            rng=np.random.default_rng(p['noise_rng_base']+bank)
            raw['source_bank'].append(np.full(600,bank,np.int64));raw['uniforms'].append(1-rng.random((600,81)))
            raw['offsets'].append(rng.integers(1,9,(600,81),dtype=np.int64))
        raw={key:np.concatenate(values) for key,values in raw.items()}
        noisy,cells=inherited.corrupt(np,raw['solutions'],raw['free'],raw['uniforms'],raw['offsets'],hot,CONDITION)
        job_id=f'g{group:02d}-{CONDITION}';path=OUT/'inputs'/f'{job_id}.npz';path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,**raw,noisy=noisy,cells=cells)
        entries.append(dict(job_id=job_id,group=group,condition=CONDITION,banks=banks,hot_cells=hot.tolist(),path=str(path.relative_to(ROOT)),sha256=sha(path)))
        jobs.append(dict(job_id=job_id,bank=group,condition=CONDITION,status='pending'))
    save(OUT/'inputs.json',dict(protocol_sha256=sha(PROTOCOL),hot_cells_by_group=heatmaps,entries=entries))
    save(OUT/'queue.json',jobs);save(OUT/'preparation.json',dict(wall_seconds=time.monotonic()-start,entries=16,new_source_rows=0))
    print(json.dumps(dict(entries=16,preparation_wall_seconds=time.monotonic()-start)))


def select_graphs():
    import numpy as np
    from regularity_gap import select
    p=json.loads(PROTOCOL.read_text());verify(p);start=time.monotonic()
    source=json.loads((OUT/'source-audit.json').read_text())
    assert source['status']=='validated' and source['protocol_sha256']==sha(PROTOCOL) and source['input_manifest_sha256']==sha(OUT/'inputs.json')
    manifest=json.loads((OUT/'inputs.json').read_text());queue=json.loads((OUT/'queue.json').read_text())
    assert len(queue)==16 and all(j['status']=='complete' for j in queue)
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
    np.savez_compressed(destination/'supports.npz',supports=np.stack(supports),groups=np.arange(16))
    save(destination/'receipt.json',dict(status='complete',protocol_sha256=sha(PROTOCOL),rows=rows,supports_sha256=sha(destination/'supports.npz'),wall_seconds=time.monotonic()-start,selector_sha256=sha(HERE/'regularity_gap.py')))
    print(json.dumps(dict(status='complete',groups=16,selection_wall_seconds=time.monotonic()-start)))


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
